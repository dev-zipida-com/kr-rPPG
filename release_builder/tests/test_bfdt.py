import importlib.util
from pathlib import Path

import numpy as np
import pytest

import krrppg_release.bfdt as bfdt_module
from krrppg_release.bfdt import (
    bfdt_frame,
    bfdt_frame_reference,
    block_means,
    check_release_setting,
    is_block_constant,
    kept_coefficients,
    release_keep_ratio,
    require_dc_only,
)

# Independent reference implementation shipped at <repo>/privacy/bfdt_transform.py
REFERENCE = Path(__file__).resolve().parents[2] / "privacy" / "bfdt_transform.py"


def _load_reference():
    if not REFERENCE.exists():
        pytest.skip("reference BFDT implementation not available")
    spec = importlib.util.spec_from_file_location("bfdt_reference", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_frame(shape=(48, 64, 3), seed=0):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def test_release_setting_keeps_only_the_dc_coefficient():
    assert kept_coefficients(8, 0.02) == 1
    require_dc_only(8, 0.02)


def test_settings_keeping_more_than_dc_are_rejected():
    with pytest.raises(ValueError):
        require_dc_only(8, 0.1)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_invalid_keep_ratio_is_rejected(ratio):
    with pytest.raises(ValueError):
        kept_coefficients(8, ratio)


def test_each_block_becomes_its_rounded_mean_colour():
    frame = _random_frame((16, 24, 3), seed=1)
    out = bfdt_frame(frame)
    expected = np.rint(frame[8:16, 16:24].astype(np.float64).mean(axis=(0, 1))).astype(np.uint8)
    assert np.array_equal(out[8:16, 16:24], np.broadcast_to(expected, (8, 8, 3)))
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_matches_reference_bfdt_implementation_up_to_integer_rounding():
    reference = _load_reference()
    frame = _random_frame(seed=2)
    expected = reference.apply_bfdt(frame.astype(np.float32), block_size=8, keep_ratio=0.02)
    ours = bfdt_frame(frame).astype(np.float64)
    assert np.max(np.abs(ours - expected)) <= 0.5 + 1e-3


def test_output_is_block_constant_and_random_input_is_not():
    frame = _random_frame(seed=3)
    assert not is_block_constant(frame)
    assert is_block_constant(bfdt_frame(frame))


def test_transform_is_idempotent_and_keeps_flat_frames():
    flat = np.full((8, 16, 3), 77, dtype=np.uint8)
    assert np.array_equal(bfdt_frame(flat), flat)
    once = bfdt_frame(_random_frame(seed=4))
    assert np.array_equal(bfdt_frame(once), once)


def test_half_way_means_round_to_even():
    block = np.zeros((8, 8, 3), dtype=np.uint8)
    block[:4] = 1  # mean 0.5 -> 0
    block[:4, :, 1] = 2
    block[4:, :, 1] = 1  # mean 1.5 -> 2
    means = block_means(block)
    assert means[0, 0, 0] == 0
    assert means[0, 0, 1] == 2


def test_block_means_shape():
    assert block_means(_random_frame()).shape == (6, 8, 3)


def test_frame_size_must_be_a_multiple_of_the_block_size():
    with pytest.raises(ValueError):
        bfdt_frame(np.zeros((10, 16, 3), dtype=np.uint8))


def test_non_uint8_frames_are_rejected():
    with pytest.raises(ValueError):
        bfdt_frame(np.zeros((8, 8, 3), dtype=np.float32))


def test_fast_path_matches_reference_for_every_possible_block_sum():
    probe = bfdt_module._all_block_sums_image(8)
    sums = probe.reshape(probe.shape[0] // 8, 8, probe.shape[1] // 8, 8, 3).sum(axis=(1, 3))
    assert set(range(64 * 255 + 1)) <= set(np.unique(sums).tolist())
    assert bfdt_module._fast_path_ok(8)
    assert np.array_equal(bfdt_frame(probe), bfdt_frame_reference(probe))


@pytest.mark.parametrize("block, ratio", [(8, 0.02), (16, 1 / 256), (32, 1 / 1024)])
def test_release_keep_ratio_keeps_only_the_dc_coefficient(block, ratio):
    assert release_keep_ratio(block) == ratio
    assert kept_coefficients(block, ratio) == 1


def test_manuscript_keep_ratio_is_not_dc_only_at_16px():
    assert kept_coefficients(16, 0.02) == 5
    with pytest.raises(ValueError):
        require_dc_only(16, 0.02)


def test_16px_blocks_become_their_rounded_mean_colour():
    frame = _random_frame(seed=7)
    out = bfdt_frame(frame, 16, release_keep_ratio(16))
    expected = np.rint(frame[16:32, 32:48].astype(np.float64).mean(axis=(0, 1))).astype(np.uint8)
    assert np.array_equal(out[16:32, 32:48], np.broadcast_to(expected, (16, 16, 3)))
    assert np.array_equal(out, bfdt_frame_reference(frame, 16))
    assert is_block_constant(out, 16)
    assert not is_block_constant(bfdt_frame(frame), 16)  # a block-8 frame is not block-constant at 16


def test_fast_path_matches_reference_for_every_possible_16px_block_sum():
    probe = bfdt_module._all_block_sums_image(16)
    sums = probe.reshape(probe.shape[0] // 16, 16, probe.shape[1] // 16, 16, 3).sum(axis=(1, 3))
    assert set(range(256 * 255 + 1)) <= set(np.unique(sums).tolist())
    assert bfdt_module._fast_path_ok(16)


def test_release_settings_are_accepted():
    check_release_setting(8, 0.02, (640, 480))
    check_release_setting(16, 1 / 256, (640, 480))
    check_release_setting(16, 1 / 256, None)


@pytest.mark.parametrize("block", [1, 2, 4, 5, 10, 32])
def test_block_sizes_outside_the_release_list_are_rejected(block):
    # block 1 is the identity: DC-only and block-constant, yet the raw face
    with pytest.raises(ValueError, match="block size"):
        check_release_setting(block, release_keep_ratio(block), (640, 480))


@pytest.mark.parametrize("block, ratio, size", [
    (16, 0.02, (640, 480)),     # keeps 5 coefficients
    (16, 1 / 256, (40, 48)),    # 40 is not a multiple of 16
    (0, 0.02, (640, 480)),
])
def test_unusable_release_settings_are_rejected(block, ratio, size):
    with pytest.raises(ValueError):
        check_release_setting(block, ratio, size)


def test_reference_path_is_used_when_fast_path_is_not_trusted(monkeypatch):
    frame = _random_frame(seed=6)
    expected = bfdt_frame(frame)
    monkeypatch.setitem(bfdt_module._FAST_PATH_OK, 8, False)
    assert np.array_equal(bfdt_frame(frame), expected)
