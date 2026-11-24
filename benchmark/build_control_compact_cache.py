#!/usr/bin/env python3
"""Build the matched untransformed-video crop cache for the paired utility control (Table 10).

Everything except the pixels is shared with build_compact_cache.py: the released BVP, the released
face boxes (session median, 1.5 enlargement), the 180-frame skip, the area resize and the session
statistics. A model trained on this cache is therefore the matched control for a model trained on
the cache of the released, de-identified videos.

--video-map is a private CSV with the columns ``session,video_path``. Frame i of each listed video
must be the source of frame i of the released video of that session. Before a session is accepted
its source video is checked against the released video: same frame count and frame size, and, at a
few frames spread over the recording, the same picture on a coarse grid (a released frame is a
block average of its source frame) but not the identical frame (the source must not be the released
video itself). For the 8-pixel release all 1,166 true pairs differ by 0.13-0.17 grey levels on that
grid, a different recording of the same person by about 2 or more and a recording of another person
by more than 14; the limit is 0.5. A shift of a few frames is not detected by this comparison.
Source paths are never copied into the cache or the log.

The cache holds face crops of untransformed recordings. It is private derived data: keep it outside
this repository, do not distribute it, and remove it when the control runs are finished.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from build_compact_cache import SKIP, crop_box, digest, session_stats

ARM = "untransformed_control"
ALLOWED_SIZES = (36, 72, 128)
RELEASED_VIDEO = "video_bfdt_ffv1.avi"
REPO = Path(__file__).resolve().parents[1]
MANIFEST_FIELDS = ("session", "participant_key", "task", "frames", "start_frame", "common_eval_frames")
CHECK_CELL = 16  # pixels per cell of the coarse grid used to compare source and released frames
MAX_MEAN_ABS_DIFF = 0.5  # grey levels on that grid; measured values are given in the module docstring
HIDDEN_PATH = "<source video>"
LOGGER = logging.getLogger("build_control_compact_cache")

Job = tuple[str, str, dict[str, str], str, tuple[int, ...], dict[str, Any], int]


class InputError(ValueError):
    """A problem with the command-line inputs; reported without a traceback."""


@dataclass(frozen=True)
class Session:
    """What is known about a session before any frame is read."""

    key: str
    folder: Path  # released session folder
    bvp: np.ndarray
    box_rows: int
    valid_box_rows: int
    roi: list[int]
    header_frames: int
    frame_shape: tuple[int, int]
    frames: int  # frames kept after the skip


def video_header(path: Path) -> tuple[int, int, int]:
    """Frame count, width and height from the container header; no frame is read."""
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not capture.isOpened():
            raise ValueError("video cannot be opened")
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def matching_header(key: str, video_path: str, released_path: Path) -> tuple[int, int, int]:
    """Header of the released video, after checking that the source video has the same one."""
    try:
        released = video_header(released_path)
    except ValueError:
        raise ValueError(f"The released video of {key} cannot be opened") from None
    try:
        is_released_file = os.path.samefile(video_path, released_path)
        control = None if is_released_file else video_header(Path(video_path))
    except (OSError, ValueError):
        raise ValueError(f"The source video of {key} cannot be opened") from None
    if is_released_file:
        raise ValueError(f"The source video of {key} is the released video itself, not the untransformed recording")
    if control != released:
        raise ValueError(
            f"The source video of {key} does not match the released video "
            f"(frames, width, height): {control} vs {released}"
        )
    return released


def coarse(frame: np.ndarray) -> np.ndarray:
    """Area-averaged picture on a grid of CHECK_CELL-pixel cells."""
    height, width = frame.shape[:2]
    size = (max(1, width // CHECK_CELL), max(1, height // CHECK_CELL))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA).astype(np.float32)


def check_indices(first: int, count: int, how_many: int) -> list[int]:
    """Frame indices spread evenly over first..first+count-1."""
    if how_many <= 0 or count <= 0:
        return []
    if how_many == 1 or count == 1:
        return [first]
    return sorted({first + round(step * (count - 1) / (how_many - 1)) for step in range(how_many)})


def read_crops(
    video_path: str,
    roi: list[int],
    first: int,
    count: int,
    crops: dict[int, np.ndarray],
    frame_shape: tuple[int, int],
    keep: set[int],
) -> tuple[int, dict[int, np.ndarray]]:
    """Read the whole video once; store the resized face crop of frames first..first+count-1.

    Returns the number of frames read and a copy of every frame whose index is listed in keep.
    """
    x1, y1, x2, y2 = roi
    kept: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != frame_shape:
                raise ValueError(f"frame {index} has an unexpected size")
            if first <= index < first + count:
                face = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                for size, array in crops.items():
                    array[index - first] = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
            if index in keep:
                kept[index] = frame.copy()
            index += 1
    finally:
        capture.release()
    return index, kept


def released_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """The released video at the given frame indices."""
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if indices and not capture.isOpened():
            raise ValueError("released video cannot be opened")
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"released frame {index} cannot be read")
            frames[index] = frame
    finally:
        capture.release()
    return frames


def source_difference(key: str, source: dict[int, np.ndarray], released: dict[int, np.ndarray]) -> float:
    """Largest coarse-grid difference over the compared frames; raises unless the two videos belong together."""
    if not released:
        return 0.0
    if all(np.array_equal(source[index], frame) for index, frame in released.items()):
        raise ValueError(f"The source video of {key} is the released video itself, not the untransformed recording")
    worst = max(float(np.mean(np.abs(coarse(source[index]) - coarse(frame)))) for index, frame in released.items())
    if worst > MAX_MEAN_ABS_DIFF:
        raise ValueError(
            f"The source video of {key} does not match the released video content "
            f"(mean absolute difference {worst:.1f} grey levels on a {CHECK_CELL}-pixel grid)"
        )
    return worst


def discard_partial(dest: Path) -> None:
    """Remove crops of a session that did not finish; a finished session (stats.json) is kept."""
    if not dest.is_dir() or (dest / "stats.json").exists():
        return
    for leftover in list(dest.glob("crop_*.npy")) + [dest / "bvp.npy", dest / "stats.json.tmp"]:
        leftover.unlink(missing_ok=True)


def source_signature(video_path: str, key: str) -> dict[str, int]:
    try:
        status = os.stat(video_path)
    except OSError:
        raise ValueError(f"The source video of {key} cannot be opened") from None
    return {"source_size_bytes": status.st_size, "source_mtime_ns": status.st_mtime_ns}


def reusable(dest: Path, previous: dict[str, Any], provenance: dict[str, Any], expected: int,
             signature: dict[str, int], sizes: tuple[int, ...]) -> bool:
    """A finished session is reused only if it was built from the same inputs and its files are present."""
    arrays = [dest / f"crop_{size}.npy" for size in sizes]
    packed = [dest / f"crop_{size}{suffix}" for size in sizes for suffix in (".zlib", ".zlib.json")]
    present = (dest / "bvp.npy").exists() and (
        all(path.exists() for path in arrays) or all(path.exists() for path in packed))
    same_source = all(previous.get("inputs", {}).get(name) == value for name, value in signature.items())
    return (previous.get("cache_provenance") == provenance and previous.get("frames") == expected
            and same_source and present)


def plan_session(root: str, row: dict[str, str], video_path: str) -> Session:
    """Read the released signals and check the source video header before any crop is written."""
    key = row["session"]
    folder = Path(root) / row["participant_key"] / row["task"]
    bvp = np.genfromtxt(folder / "BVP_synchronized.csv", delimiter=",", names=True)["bvp"]
    boxes = np.genfromtxt(folder / "face_bboxes.csv", delimiter=",", skip_header=1, ndmin=2)
    header, width, height = matching_header(key, video_path, folder / RELEASED_VIDEO)
    roi, valid = crop_box(boxes, width, height)
    frames = min(len(bvp), header) - SKIP
    if frames != int(row["available_frames_after_skip"]):
        raise ValueError(f"Eligibility length mismatch {key}: {frames}")
    return Session(key, folder, bvp, len(boxes), valid, roi, header, (height, width), frames)


def fill_crops(session: Session, video_path: str, dest: Path, sizes: tuple[int, ...], indices: list[int],
               signature: dict[str, int]) -> tuple[dict[int, np.ndarray], float]:
    """Write the crops of every size; returns them with the largest source/released difference."""
    dest.mkdir(parents=True, exist_ok=True)
    crops = {
        size: np.lib.format.open_memmap(
            dest / f"crop_{size}.npy", mode="w+", dtype=np.uint8, shape=(session.frames, size, size, 3)
        )
        for size in sizes
    }
    try:
        frames_read, kept = read_crops(
            video_path, session.roi, SKIP, session.frames, crops, session.frame_shape, set(indices))
    except ValueError as exc:
        raise ValueError(f"The source video of {session.key} cannot be used: {exc}") from None
    if frames_read != session.header_frames:
        raise ValueError(
            f"Frames read differ from the header count for {session.key}: {frames_read}/{session.header_frames}")
    if source_signature(video_path, session.key) != signature:
        raise ValueError(f"The source video of {session.key} changed while it was read")
    worst = source_difference(session.key, kept, released_frames(session.folder / RELEASED_VIDEO, indices))
    for array in crops.values():
        array.flush()
    return crops, worst


def build_session(job: Job, dest: Path, signature: dict[str, int]) -> dict[str, Any]:
    root, _, row, video_path, sizes, provenance, verify_frames = job
    started = time.time()
    session = plan_session(root, row, video_path)
    indices = check_indices(SKIP, session.frames, verify_frames)
    crops, worst = fill_crops(session, video_path, dest, sizes, indices, signature)
    waveform = np.asarray(session.bvp[SKIP:SKIP + session.frames], dtype=np.float64)
    np.save(dest / "bvp.npy", waveform)
    stats = {
        "schema_version": "compact-v1",
        "arm": ARM,
        "session": session.key,
        "participant_key": row["participant_key"],
        "task": row["task"],
        "frames": session.frames,
        "decoded_frames": session.header_frames,
        "video_frames_header": session.header_frames,
        "bvp_rows": len(session.bvp),
        "skip_frames": SKIP,
        "common_eval_frames": int(row["common_eval_frames"]),
        "roi": session.roi,
        "bbox_rows": session.box_rows,
        "valid_bbox_rows": session.valid_box_rows,
        "bbox_policy": "median of all finite valid released boxes; 1.5 enlargement",
        "sizes": {str(size): session_stats(array) for size, array in crops.items()},
        "label": {
            "mean": float(waveform.mean()),
            "std": float(waveform.std()),
            "diff_std": float(np.diff(waveform).std()),
        },
        "inputs": {
            "relative_session": f"{row['participant_key']}/{row['task']}",
            "bvp_sha256": digest(session.folder / "BVP_synchronized.csv"),
            "bbox_sha256": digest(session.folder / "face_bboxes.csv"),
            "source_frame_index_mapping": "identity_zero_based_decode_order",
            **signature,
        },
        "source_check": {
            "frames": indices,
            "grid_cell_pixels": CHECK_CELL,
            "max_mean_abs_diff": round(worst, 4),
            "limit": MAX_MEAN_ABS_DIFF,
        },
        "moment_precision": "float32 frame/difference arithmetic; float64 accumulated population moments",
        "cache_provenance": provenance,
        "seconds": round(time.time() - started, 3),
    }
    complete = dest / "stats.json"
    temporary = complete.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(stats, indent=2) + "\n")
    temporary.replace(complete)
    return {"session": session.key, "status": "created", "frames": session.frames, "seconds": stats["seconds"]}


def build_one(job: Job) -> dict[str, Any]:
    """Build the crops of one session. Error messages name the session, never the source path."""
    _, out, row, video_path, sizes, provenance, _ = job
    cv2.setNumThreads(1)
    key = row["session"]
    dest = Path(out) / key
    expected = int(row["available_frames_after_skip"])
    signature = source_signature(video_path, key)
    complete = dest / "stats.json"
    if complete.exists():
        if reusable(dest, json.loads(complete.read_text()), provenance, expected, signature, sizes):
            return {"session": key, "status": "reused", "frames": expected, "seconds": 0}
        raise RuntimeError(f"Existing cache is incomplete or was built from different inputs: {key}")
    try:
        return build_session(job, dest, signature)
    except Exception:
        discard_partial(dest)
        raise


def read_video_map(path: Path) -> dict[str, str]:
    """session -> source video path; a session may be listed only once and needs a path."""
    mapping: dict[str, str] = {}
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                session, video_path = row["session"], row["video_path"]
                if session in mapping:
                    raise InputError(f"Cannot use --video-map: session listed twice: {session}")
                if not video_path:
                    raise InputError(f"Cannot use --video-map: empty video_path: {session}")
                mapping[session] = video_path
    except (OSError, KeyError, UnicodeDecodeError, csv.Error):
        raise InputError(
            "Cannot use --video-map: expected a readable CSV with the columns session,video_path") from None
    return mapping


def non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="released KR-rPPG tree (BVP, boxes, videos)")
    parser.add_argument("--video-map", type=Path, required=True, help="private CSV: session,video_path")
    parser.add_argument("--out", type=Path, required=True, help="private cache directory outside the repository")
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", choices=ALLOWED_SIZES, default=list(ALLOWED_SIZES))
    parser.add_argument("--verify-frames", type=non_negative, default=3,
                        help="frames per session compared with the released video (0 switches the check off)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sessions", nargs="+", help="build only these sessions (no manifest is written)")
    return parser.parse_args(argv)


def is_truthy(value: str) -> bool:
    return str(value).lower() in {"true", "1"}


def select_rows(eligibility: Path, sessions: list[str] | None) -> list[dict[str, str]]:
    with eligibility.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if is_truthy(row["eligible"])]
    known = {row["session"] for row in rows}
    if len(known) != len(rows):
        raise InputError("The eligibility file lists a session more than once")
    if not sessions:
        return rows
    unknown = [session for session in sessions if session not in known]
    if unknown:
        raise InputError(f"{len(unknown)} requested session(s) are unknown or not eligible (first: {unknown[0]})")
    return [row for row in rows if row["session"] in set(sessions)]


def load_inputs(args: argparse.Namespace) -> tuple[Path, list[dict[str, str]], dict[str, str]]:
    """Cache folder, eligible rows and the session -> source video map, or InputError."""
    out = args.out.resolve()
    if out == REPO or REPO in out.parents:
        raise InputError("The cache holds private face crops and must be outside the repository")
    rows = select_rows(args.eligibility, args.sessions)
    video_map = read_video_map(args.video_map)
    missing = [row["session"] for row in rows if row["session"] not in video_map]
    if missing:
        raise InputError(
            f"{len(missing)} eligible session(s) have no source video in --video-map (first: {missing[0]})")
    return out, rows, video_map


def build_all(jobs: dict[str, Job], workers: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build every session; a failure is recorded per session and never shows the source path."""
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(build_one, job): session for session, job in jobs.items()}
        for done, future in enumerate(as_completed(futures), 1):
            session = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as exc:  # reported per session; the run fails after all sessions were tried
                message = str(exc).replace(jobs[session][3], HIDDEN_PATH)
                result = {"session": session, "status": "error", "error": message}
                errors.append(result)
            LOGGER.info(json.dumps({"done": done, "total": len(jobs), **result}))
    return results, errors


def write_manifest(out: Path, rows: list[dict[str, str]]) -> None:
    with (out / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "session": row["session"],
                "participant_key": row["participant_key"],
                "task": row["task"],
                "frames": row["available_frames_after_skip"],
                "start_frame": SKIP,
                "common_eval_frames": row["common_eval_frames"],
            })


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.umask(0o077)
    args = parse_args(argv)
    try:
        out, rows, video_map = load_inputs(args)
    except InputError as exc:
        LOGGER.error("%s.", exc)
        return 2
    sizes = tuple(sorted(set(args.sizes)))
    provenance = {
        "arm": ARM,
        "sizes": list(sizes),
        "verify_frames": args.verify_frames,
        "video_map_sha256": digest(args.video_map),
        "release_manifest_sha256": digest(args.release_manifest),
        "eligibility_sha256": digest(args.eligibility),
        "builder_sha256": digest(Path(__file__)),
        "shared_crop_and_stats_implementation_sha256": digest(Path(__file__).with_name("build_compact_cache.py")),
    }
    out.mkdir(parents=True, exist_ok=True)
    is_full_build = not args.sessions
    if is_full_build:
        (out / "manifest.csv").unlink(missing_ok=True)  # written again only when every session is complete
    started = time.time()
    jobs: dict[str, Job] = {
        row["session"]: (str(args.root), str(out), row, video_map[row["session"]], sizes, provenance,
                         args.verify_frames)
        for row in rows
    }
    results, errors = build_all(jobs, args.workers)
    report = {
        "requested_sessions": len(rows),
        "completed_sessions": len(results),
        "errors": errors,
        "wall_seconds": round(time.time() - started, 3),
        "workers": args.workers,
        "provenance": provenance,
    }
    suffix = "all" if is_full_build else "subset"
    (out / f"build_report_{suffix}.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        LOGGER.error("%d session(s) failed; no manifest written.", len(errors))
        return 1
    if is_full_build:
        write_manifest(out, rows)
    summary = {key: report[key] for key in ("requested_sessions", "completed_sessions", "wall_seconds")}
    sys.stdout.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
