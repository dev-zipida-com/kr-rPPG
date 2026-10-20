"""Release-format BFDT (block-wise frequency-domain transform, Huang et al. 2025).

The reference implementation (privacy/bfdt_transform.py in this repository)
keeps `max(1, int(block_size**2 * keep_ratio))` zig-zag DCT coefficients per block.
At the release setting (8x8 blocks, k_r = 0.02) that is a single coefficient, the DC
term, and the orthonormal inverse DCT of a DC-only block is the block mean. Means are
computed exactly and rounded to the nearest integer (ties to even) for 8-bit storage.
The optional 16 x 16 block size uses the DC-only k_r from `release_keep_ratio` (1/256).

`bfdt_frame` uses OpenCV (INTER_AREA down, INTER_NEAREST up), about 30x faster than
numpy. It is used only after `_fast_path_ok` has confirmed, once per process, that it
reproduces the numpy definition for every possible block sum; otherwise numpy is used.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .constants import BLOCK_SIZE, KEEP_RATIO, RELEASE_BLOCK_SIZES

_FAST_PATH_OK: Dict[int, bool] = {}


def kept_coefficients(block_size: int, keep_ratio: float) -> int:
    """Number of DCT coefficients kept per block (same rule as the reference code)."""
    if block_size < 1:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    return max(1, int(block_size * block_size * keep_ratio))


def require_dc_only(block_size: int, keep_ratio: float) -> None:
    kept = kept_coefficients(block_size, keep_ratio)
    if kept != 1:
        raise ValueError(
            f"block_size={block_size}, keep_ratio={keep_ratio} keeps {kept} coefficients; "
            f"only the DC-only release setting is supported (keep_ratio {release_keep_ratio(block_size)})"
        )


def release_keep_ratio(block_size: int) -> float:
    """DC-only k_r for a block size: the manuscript's 0.02 where that already keeps one
    coefficient (block size 8), otherwise exactly 1/b^2 (block size 16: 0.02 would keep 5)."""
    if kept_coefficients(block_size, KEEP_RATIO) == 1:
        return KEEP_RATIO
    return 1.0 / (block_size * block_size)


def check_release_setting(block_size: int, keep_ratio: float, frame_size: Optional[Tuple[int, int]]) -> None:
    """Fail fast on settings that must not be released or that no session could be built with."""
    if block_size not in RELEASE_BLOCK_SIZES:
        raise ValueError(f"block size {block_size} px is not a release block size; use one of {RELEASE_BLOCK_SIZES}")
    require_dc_only(block_size, keep_ratio)
    if frame_size is not None and any(side % block_size for side in frame_size):
        raise ValueError(f"frame size {frame_size[0]}x{frame_size[1]} is not a multiple of the {block_size}px block size")


def _blocks(frame: np.ndarray, block_size: int) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3:
        raise ValueError(f"expected an HxWxC uint8 frame, got {frame.dtype} with shape {frame.shape}")
    height, width, channels = frame.shape
    if height % block_size or width % block_size:
        raise ValueError(f"frame size {width}x{height} is not a multiple of the {block_size}px block size")
    return frame.reshape(height // block_size, block_size, width // block_size, block_size, channels)


def block_means(frame: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Reference definition: rounded per-block mean colour, shape (H/b, W/b, C), uint8."""
    sums = _blocks(frame, block_size).sum(axis=(1, 3), dtype=np.uint32)
    return np.rint(sums / float(block_size * block_size)).astype(np.uint8)


def _expand(means: np.ndarray, block_size: int) -> np.ndarray:
    rows, cols, channels = means.shape
    tiled = np.broadcast_to(means[:, None, :, None, :], (rows, block_size, cols, block_size, channels))
    return np.ascontiguousarray(tiled).reshape(rows * block_size, cols * block_size, channels)


def bfdt_frame_reference(frame: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    return _expand(block_means(frame, block_size), block_size)


def _bfdt_frame_fast(frame: np.ndarray, block_size: int) -> np.ndarray:
    height, width = frame.shape[:2]
    means = cv2.resize(frame, (width // block_size, height // block_size), interpolation=cv2.INTER_AREA)
    if means.ndim == 2:
        means = means[:, :, None]
    return cv2.resize(means, (width, height), interpolation=cv2.INTER_NEAREST).reshape(frame.shape)


def _all_block_sums_image(block_size: int) -> np.ndarray:
    """3-channel image whose blocks cover every possible block sum 0..b*b*255."""
    n_pixels = block_size * block_size
    sums = np.arange(n_pixels * 255 + 1)
    base, extra = sums // n_pixels, sums % n_pixels
    blocks = (base[:, None] + (np.arange(n_pixels)[None, :] < extra[:, None])).astype(np.uint8)
    side = int(np.ceil(np.sqrt(len(sums))))
    padded = np.concatenate([blocks, np.zeros((side * side - len(sums), n_pixels), np.uint8)])
    plane = padded.reshape(side, side, block_size, block_size).transpose(0, 2, 1, 3)
    plane = plane.reshape(side * block_size, side * block_size)
    shifted = np.roll(plane, block_size, axis=1)
    return np.ascontiguousarray(np.stack([plane, shifted, np.roll(plane, block_size, axis=0)], axis=-1))


def _fast_path_ok(block_size: int) -> bool:
    if block_size not in _FAST_PATH_OK:
        probe = _all_block_sums_image(block_size)
        _FAST_PATH_OK[block_size] = bool(
            np.array_equal(_bfdt_frame_fast(probe, block_size), bfdt_frame_reference(probe, block_size))
        )
    return _FAST_PATH_OK[block_size]


def bfdt_frame(frame: np.ndarray, block_size: int = BLOCK_SIZE, keep_ratio: float = KEEP_RATIO) -> np.ndarray:
    """Apply release-format BFDT to one BGR uint8 frame."""
    require_dc_only(block_size, keep_ratio)
    _blocks(frame, block_size)  # validates dtype and shape
    if frame.shape[2] in (1, 3, 4) and _fast_path_ok(block_size):
        return _bfdt_frame_fast(np.ascontiguousarray(frame), block_size)
    return bfdt_frame_reference(frame, block_size)


def is_block_constant(frame: np.ndarray, block_size: int = BLOCK_SIZE) -> bool:
    """True when every block holds a single colour (the DC-only BFDT signature)."""
    blocks = _blocks(frame, block_size)
    return bool(np.all(blocks == blocks[:, :1, :, :1, :]))
