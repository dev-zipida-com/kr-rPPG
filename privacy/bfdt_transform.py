"""
Block-wise Frequency Domain Transformation (BFDT) for face de-identification.

Based on: Huang et al., "Privacy-preserving remote heart rate estimation
using Block-wise Frequency Domain Transformation", BSPC 2025.

Usage:
    from bfdt_transform import apply_bfdt, apply_bfdt_to_video
"""
import numpy as np
from scipy.fft import dctn, idctn


def _zigzag_indices(n):
    """Generate zigzag scan indices for an n x n matrix."""
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                indices.append((i, s - i))
        else:
            for i in range(max(0, s - n + 1), min(s + 1, n)):
                indices.append((i, s - i))
    return indices


def _create_frequency_mask(block_size, keep_count):
    """Create a mask keeping the lowest-frequency DCT coefficients (zigzag order)."""
    mask = np.zeros((block_size, block_size))
    zigzag = _zigzag_indices(block_size)
    for k in range(min(keep_count, len(zigzag))):
        i, j = zigzag[k]
        mask[i, j] = 1.0
    return mask


def apply_bfdt(frame, block_size=8, keep_ratio=0.5):
    """
    Apply BFDT to a single frame (vectorized).

    Args:
        frame: np.array (H, W, C), uint8 or float [0-255]
        block_size: size of each square DCT block
        keep_ratio: fraction of DCT coefficients to keep (0.0 to 1.0)

    Returns:
        np.array of same shape/dtype, BFDT-processed frame
    """
    if keep_ratio >= 1.0:
        return frame.copy()

    H, W, C = frame.shape
    total_coeffs = block_size * block_size
    keep_count = max(1, int(total_coeffs * keep_ratio))
    mask = _create_frequency_mask(block_size, keep_count)

    original_dtype = frame.dtype

    # Pad to multiple of block_size
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(frame.astype(np.float32),
                        ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded = frame.astype(np.float32)

    pH, pW = padded.shape[:2]
    nH, nW = pH // block_size, pW // block_size

    # Reshape into blocks: (pH, pW, C) -> (nH, bs, nW, bs, C) -> (nH, nW, C, bs, bs)
    blocks = padded.reshape(nH, block_size, nW, block_size, C)
    blocks = blocks.transpose(0, 2, 4, 1, 3)  # (nH, nW, C, bs, bs)
    shape_save = blocks.shape
    blocks = blocks.reshape(-1, block_size, block_size)  # (nH*nW*C, bs, bs)

    # Vectorized DCT on all blocks at once
    dct_blocks = dctn(blocks, type=2, norm='ortho', axes=(-2, -1))
    dct_blocks *= mask  # broadcast (bs, bs) mask
    reconstructed = idctn(dct_blocks, type=2, norm='ortho', axes=(-2, -1))

    # Reshape back
    reconstructed = reconstructed.reshape(shape_save)  # (nH, nW, C, bs, bs)
    reconstructed = reconstructed.transpose(0, 3, 1, 4, 2)  # (nH, bs, nW, bs, C)
    reconstructed = reconstructed.reshape(pH, pW, C)

    result = np.clip(reconstructed[:H, :W, :], 0, 255)
    if original_dtype == np.uint8:
        return result.astype(np.uint8)
    return result.astype(original_dtype)


def apply_bfdt_to_video(frames, block_size=8, keep_ratio=0.5):
    """
    Apply BFDT to all frames in a video (batch vectorized).

    Args:
        frames: np.array (T, H, W, C)
        block_size: DCT block size
        keep_ratio: fraction of coefficients to keep

    Returns:
        np.array (T, H, W, C), same dtype
    """
    if keep_ratio >= 1.0:
        return frames.copy()

    T, H, W, C = frames.shape
    total_coeffs = block_size * block_size
    keep_count = max(1, int(total_coeffs * keep_ratio))
    mask = _create_frequency_mask(block_size, keep_count)

    original_dtype = frames.dtype

    # Pad spatial dims to multiple of block_size
    pad_h = (block_size - H % block_size) % block_size
    pad_w = (block_size - W % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(frames.astype(np.float32),
                        ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded = frames.astype(np.float32)

    pH, pW = padded.shape[1], padded.shape[2]
    nH, nW = pH // block_size, pW // block_size

    # Process in batches to limit memory (~500 frames at a time)
    batch_size = 500
    result = np.empty_like(padded)

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        batch = padded[start:end]  # (B, pH, pW, C)
        B = batch.shape[0]

        # Reshape: (B, nH, bs, nW, bs, C) -> (B*nH*nW*C, bs, bs)
        blocks = batch.reshape(B, nH, block_size, nW, block_size, C)
        blocks = blocks.transpose(0, 1, 3, 5, 2, 4)  # (B, nH, nW, C, bs, bs)
        flat = blocks.reshape(-1, block_size, block_size)

        # Vectorized DCT + mask + IDCT
        dct_flat = dctn(flat, type=2, norm='ortho', axes=(-2, -1))
        dct_flat *= mask
        rec_flat = idctn(dct_flat, type=2, norm='ortho', axes=(-2, -1))

        # Reshape back: (B*nH*nW*C, bs, bs) -> (B, pH, pW, C)
        rec = rec_flat.reshape(B, nH, nW, C, block_size, block_size)
        rec = rec.transpose(0, 1, 4, 2, 5, 3)  # (B, nH, bs, nW, bs, C)
        result[start:end] = rec.reshape(B, pH, pW, C)

    result = np.clip(result[:, :H, :W, :], 0, 255)
    if original_dtype == np.uint8:
        return result.astype(np.uint8)
    return result.astype(original_dtype)
