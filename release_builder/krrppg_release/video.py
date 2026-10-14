"""Streams a source video through BFDT into a lossless FFV1 release video.

Frames are processed one at a time (no whole-video buffering). The encoded file is
written to a hidden temp name next to the destination, decoded again, and only renamed
into place when every frame is block-constant and the decoded pixels hash to exactly
what was encoded.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

from .bfdt import bfdt_frame, is_block_constant, require_dc_only
from .constants import BLOCK_SIZE, FPS, KEEP_RATIO, RELEASE_FILE_MODE
from .safety import PathPolicy, remove_output_file, replace_checked

FFV1_FOURCC = cv2.VideoWriter_fourcc(*"FFV1")


class VideoError(RuntimeError):
    """A video could not be read, encoded or verified."""


@dataclass(frozen=True)
class VideoSummary:
    frames: int
    width: int
    height: int
    fps: float
    fourcc: str
    frames_sha256: str  # sha256 over the decoded BGR uint8 frames, in order
    coarser_block_constant: bool = False  # every frame is also constant in blocks twice as large


def _open(path) -> cv2.VideoCapture:
    if not Path(path).is_file():
        raise VideoError(f"video not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoError(f"cannot open video {path}")
    return capture


def _fourcc(capture: cv2.VideoCapture) -> str:
    code = int(capture.get(cv2.CAP_PROP_FOURCC))
    return code.to_bytes(4, "little").decode("ascii", errors="replace").upper()


def _size(capture: cv2.VideoCapture) -> Tuple[int, int]:
    return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))


def _frames(capture: cv2.VideoCapture) -> Iterator[np.ndarray]:
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return
            yield frame
    finally:
        capture.release()


def source_properties(path) -> Tuple[int, int, float]:
    """(width, height, fps) reported by the container."""
    capture = _open(path)
    try:
        width, height = _size(capture)
        return width, height, float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


def bfdt_frames_sha256(path, block_size: int = BLOCK_SIZE, keep_ratio: float = KEEP_RATIO) -> Tuple[int, str]:
    """(frame count, sha256) of BFDT applied to every frame of `path`; used to cross-check releases."""
    digest = hashlib.sha256()
    count = 0
    for frame in _frames(_open(path)):
        digest.update(bfdt_frame(frame, block_size, keep_ratio).tobytes())
        count += 1
    return count, digest.hexdigest()


def inspect_release_video(path, block_size: int = BLOCK_SIZE) -> VideoSummary:
    """Decode a release video and require FFV1 with block-constant frames throughout."""
    capture = _open(path)
    fourcc = _fourcc(capture)
    width, height = _size(capture)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    digest = hashlib.sha256()
    count = 0
    coarser = 2 * block_size
    coarser_constant = width % coarser == 0 and height % coarser == 0
    for frame in _frames(capture):
        if not is_block_constant(frame, block_size):
            raise VideoError(f"{path}: frame {count} is not block-constant, so it is not DC-only BFDT")
        if coarser_constant and not is_block_constant(frame, coarser):
            coarser_constant = False
        digest.update(frame.tobytes())
        count += 1
    if fourcc != "FFV1":
        raise VideoError(f"{path}: codec {fourcc}, expected FFV1")
    if count == 0:
        raise VideoError(f"{path}: no decodable frames")
    return VideoSummary(count, width, height, fps, fourcc, digest.hexdigest(), coarser_constant)


def transcode_to_release(
    policy: PathPolicy,
    src,
    dst,
    *,
    expected_frames: Optional[int] = None,
    expected_size: Optional[Tuple[int, int]] = None,
    block_size: int = BLOCK_SIZE,
    keep_ratio: float = KEEP_RATIO,
    fps: float = FPS,
) -> VideoSummary:
    """BFDT every frame of `src` into an FFV1 file at `dst` (atomic, verified)."""
    require_dc_only(block_size, keep_ratio)
    dst = policy.check_writable(dst)
    capture = _open(src)
    width, height = _size(capture)
    if expected_size is not None and (width, height) != tuple(expected_size):
        capture.release()
        raise VideoError(f"{src}: frame size {width}x{height}, expected {expected_size[0]}x{expected_size[1]}")

    tmp = dst.with_name(f".{dst.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial{dst.suffix}")
    policy.check_writable(tmp)
    try:
        writer = cv2.VideoWriter(str(tmp), FFV1_FOURCC, fps, (width, height), True)
        if not writer.isOpened():
            raise VideoError(f"cannot open an FFV1 writer for {tmp}")
        digest = hashlib.sha256()
        count = 0
        try:
            for frame in _frames(capture):
                if frame.shape != (height, width, 3):
                    raise VideoError(f"{src}: frame {count} has shape {frame.shape}")
                released = bfdt_frame(frame, block_size, keep_ratio)
                digest.update(released.tobytes())
                writer.write(released)
                count += 1
        finally:
            writer.release()
        if expected_frames is not None and count != expected_frames:
            raise VideoError(f"{src}: decoded {count} frames, expected {expected_frames}")
        written = inspect_release_video(tmp, block_size)
        if written.frames != count or written.frames_sha256 != digest.hexdigest():
            raise VideoError(f"{tmp}: lossless read-back check failed")
        os.chmod(tmp, RELEASE_FILE_MODE)
        replace_checked(policy, tmp, dst)
    except BaseException:
        remove_output_file(policy, tmp)
        raise
    finally:
        capture.release()
    return VideoSummary(count, width, height, fps, written.fourcc, written.frames_sha256)
