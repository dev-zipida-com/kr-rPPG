import csv
import os

import cv2
import numpy as np
import pytest

from conftest import FRAME_H, FRAME_W, N_FRAMES, make_video, tree_fingerprint
from krrppg_release import constants as C
from krrppg_release.bfdt import bfdt_frame
from krrppg_release.clinical import PARTICIPANT_COLUMNS, CrfSources
from krrppg_release.metadata import build_release_metadata
from krrppg_release.safety import PathPolicy, UnsafePathError, ensure_root
from krrppg_release.video import VideoError, inspect_release_video, transcode_to_release


def _setup(fake_source):
    src = fake_source["root"]
    out = fake_source["workdir"] / "out"
    priv = fake_source["workdir"] / "private"
    policy = PathPolicy(src, [out, priv])
    ensure_root(policy, out, 0o750)
    ensure_root(policy, priv, 0o700)
    return policy, src, out


def _source_video(src):
    return src / "C" / "02-groupb" / "p901" / "T00" / C.SOURCE_VIDEO


def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


# --- video ----------------------------------------------------------------------

def test_transcode_writes_lossless_bfdt_ffv1(fake_source):
    policy, src, out = _setup(fake_source)
    dst = out / C.RELEASE_VIDEO
    summary = transcode_to_release(
        policy, _source_video(src), dst, expected_frames=N_FRAMES, expected_size=(FRAME_W, FRAME_H)
    )
    assert summary.frames == N_FRAMES
    expected = [bfdt_frame(f) for f in fake_source["frames"][("C_p901", "T00")]]
    written = _read_frames(dst)
    assert len(written) == N_FRAMES
    assert all(np.array_equal(a, b) for a, b in zip(expected, written))
    info = inspect_release_video(dst)
    assert info.fourcc == "FFV1"
    assert info.frames_sha256 == summary.frames_sha256
    assert (info.width, info.height) == (FRAME_W, FRAME_H)
    assert os.listdir(out) == [C.RELEASE_VIDEO]


def test_frame_count_mismatch_fails_and_leaves_no_partial_file(fake_source):
    policy, src, out = _setup(fake_source)
    dst = out / C.RELEASE_VIDEO
    with pytest.raises(VideoError):
        transcode_to_release(policy, _source_video(src), dst, expected_frames=N_FRAMES + 1)
    assert os.listdir(out) == []


def test_unexpected_frame_size_is_rejected_before_writing(fake_source):
    policy, src, out = _setup(fake_source)
    with pytest.raises(VideoError):
        transcode_to_release(
            policy, _source_video(src), out / C.RELEASE_VIDEO, expected_frames=N_FRAMES, expected_size=(640, 480)
        )
    assert os.listdir(out) == []


def test_destination_inside_source_is_refused(fake_source):
    policy, src, _out = _setup(fake_source)
    before = tree_fingerprint(src)
    with pytest.raises(UnsafePathError):
        transcode_to_release(policy, _source_video(src), _source_video(src).parent / C.RELEASE_VIDEO)
    assert tree_fingerprint(src) == before


def test_inspect_rejects_videos_that_are_not_block_constant(tmp_path):
    rng = np.random.default_rng(5)
    path = tmp_path / "raw.avi"
    make_video(path, [rng.integers(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(2)])
    with pytest.raises(VideoError):
        inspect_release_video(path)


def test_inspect_rejects_missing_file(tmp_path):
    with pytest.raises(VideoError):
        inspect_release_video(tmp_path / "missing.avi")


# --- metadata ---------------------------------------------------------------------------------

MAPPING = {"C_p901": "p002", "C_p902": "p003", "A_p903": "p001"}


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _build_metadata(fake_source, mapping=MAPPING, **kwargs):
    policy, src, out = _setup(fake_source)
    meta_out = out / C.RELEASE_DIRNAME / C.METADATA_DIRNAME
    crf = CrfSources.for_source(src, ild_xlsx=fake_source["ild_xlsx"])
    result = build_release_metadata(policy, src / "metadata", crf, meta_out, mapping, **kwargs)
    return result, meta_out


def test_release_metadata_carries_the_24_table6_columns(fake_source):
    result, meta_out = _build_metadata(fake_source)
    assert sorted(os.listdir(meta_out)) == sorted(C.METADATA_FILES)
    assert set(result["release_sha256"]) == set(C.METADATA_FILES)
    assert (result["written"], result["unchanged"]) == (sorted(C.METADATA_FILES), [])
    with open(meta_out / "participants.csv", newline="") as fh:
        header = next(csv.reader(fh))
    assert tuple(header) == PARTICIPANT_COLUMNS and len(header) == 24
    participants = _read(meta_out / "participants.csv")
    assert [r["participant_key"] for r in participants] == ["p001", "p002", "p003"]
    assert [r["site"] for r in participants] == ["A", "C", "C"]
    for name in C.METADATA_FILES:
        text = (meta_out / name).read_text()
        assert "C_p" not in text and "A_p" not in text and "{site}" not in text
    dictionary = _read(meta_out / "participants_dictionary.csv")
    assert [r["variable"] for r in dictionary] == list(PARTICIPANT_COLUMNS)
    assert result["quality_rule_counts"]["height_outside_120_210_flagged_unchanged"] == 1


def test_unchanged_metadata_files_are_not_rewritten(fake_source):
    _result, meta_out = _build_metadata(fake_source)
    before = {name: os.stat(meta_out / name) for name in C.METADATA_FILES}
    expected = (meta_out / "participants.csv").read_bytes()
    (meta_out / "participants.csv").write_text("participant_key,sex\np001,M\n")  # e.g. an older 6-column file
    result, _ = _build_metadata(fake_source, only_changed=True)
    assert result["written"] == ["participants.csv"]
    assert result["unchanged"] == sorted(set(C.METADATA_FILES) - {"participants.csv"})
    assert (meta_out / "participants.csv").read_bytes() == expected
    for name in result["unchanged"]:
        after = os.stat(meta_out / name)
        assert (after.st_ino, after.st_mtime_ns) == (before[name].st_ino, before[name].st_mtime_ns)


def test_availability_and_folds_are_rekeyed(fake_source):
    _result, meta_out = _build_metadata(fake_source)
    availability = {r["participant_key"]: r for r in _read(meta_out / "task_session_availability.csv")}
    assert [availability["p003"][f"t0{i}_released"] for i in range(4)] == ["1", "0", "0", "0"]
    assert [r["participant_key"] for r in _read(meta_out / "fold_assignment.csv")] == ["p001", "p002", "p003"]
    dictionary = {r["variable"]: r for r in _read(meta_out / "participants_dictionary.csv")}
    assert dictionary["participant_key"]["description"] == "Pseudonymized participant identifier (pNNN; e.g., p001)"


def test_metadata_with_unmapped_participant_is_rejected(fake_source):
    partial = {k: v for k, v in MAPPING.items() if k != "A_p903"}
    with pytest.raises(ValueError):
        _build_metadata(fake_source, partial)
