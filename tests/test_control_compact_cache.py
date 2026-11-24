"""build_control_compact_cache.py: same crops and statistics as the released cache, pixels from another video."""
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "build_control_compact_cache.py"
RELEASED_BUILDER = REPO / "benchmark" / "build_compact_cache.py"
sys.path.insert(0, str(REPO / "benchmark"))
import build_control_compact_cache as builder  # noqa: E402
from build_compact_cache import SKIP, crop_box, session_stats  # noqa: E402

WIDTH, HEIGHT, FRAMES = 96, 64, SKIP + 12
BOX = (30, 20, 62, 52)
CELL = 16  # the synthetic picture is constant over CELL x CELL squares, plus fine noise
BLOCK = 8  # the released stand-in replaces every BLOCK x BLOCK square by its mean
SESSION = "demo_T00"
SAME_RECORDING, OTHER_RECORDING = 1, 2


@dataclass(frozen=True)
class Inputs:
    root: Path
    eligibility: Path
    video_map: Path
    release_manifest: Path
    control: Path
    frames: list


def source_frames(seed: int, frames: int = FRAMES, size: tuple[int, int] = (WIDTH, HEIGHT)) -> list:
    """Frames with a coarse random picture plus fine noise; different seeds give different pictures."""
    rng = np.random.default_rng(seed)
    width, height = size
    result = []
    for _ in range(frames):
        picture = rng.integers(30, 226, size=(height // CELL, width // CELL, 3))
        fine = rng.integers(-20, 21, size=(height, width, 3))
        result.append((np.kron(picture, np.ones((CELL, CELL, 1), dtype=np.int64)) + fine).astype(np.uint8))
    return result


def block_mean(frame: "np.ndarray") -> "np.ndarray":
    """Stand-in for the released transform: every BLOCK x BLOCK square is replaced by its mean."""
    height, width = frame.shape[:2]
    cells = frame.reshape(height // BLOCK, BLOCK, width // BLOCK, BLOCK, 3).mean(axis=(1, 3))
    return np.kron(np.rint(cells).astype(np.uint8), np.ones((BLOCK, BLOCK, 1), dtype=np.uint8))


def write_video(path: Path, frames: list) -> None:
    """Lossless synthetic video (frames are BGR)."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30, (width, height))
    if not writer.isOpened():
        pytest.skip("FFV1 writer unavailable")
    for frame in frames:
        writer.write(frame)
    writer.release()


def write_video_map(
    path: Path, rows: list[tuple[str, str]], header: tuple[str, str] = ("session", "video_path")
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def make_inputs(
    tmp_path: Path,
    control_seed: int = SAME_RECORDING,
    control_frames: int = FRAMES,
    control_size: tuple[int, int] = (WIDTH, HEIGHT),
) -> Inputs:
    """One released session (block mean of recording 1) and one private source video (recording control_seed)."""
    root = tmp_path / "release" / "KR-rPPG"
    session = root / "demo" / "T00"
    session.mkdir(parents=True)
    write_video(session / "video_bfdt_ffv1.avi", [block_mean(frame) for frame in source_frames(SAME_RECORDING)])
    with (session / "BVP_synchronized.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "bvp"])
        for index in range(FRAMES):
            writer.writerow([index / 30, float(np.sin(index / 5))])
    with (session / "face_bboxes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x1", "y1", "x2", "y2"])
        for _ in range(FRAMES):
            writer.writerow(BOX)
    control = tmp_path / "private" / "source.avi"
    control.parent.mkdir()
    frames = source_frames(control_seed, control_frames, control_size)
    write_video(control, frames)
    eligibility = tmp_path / "eligibility_manifest.csv"
    with eligibility.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "participant_key", "task", "eligible", "available_frames_after_skip",
                         "common_eval_frames"])
        writer.writerow([SESSION, "demo", "T00", "1", FRAMES - SKIP, FRAMES - SKIP])
    video_map = tmp_path / "private" / "video_map.csv"
    write_video_map(video_map, [(SESSION, str(control))])
    release_manifest = tmp_path / "release_manifest.json"
    release_manifest.write_text("{}\n")
    return Inputs(root, eligibility, video_map, release_manifest, control, frames)


def run(inputs: Inputs, out: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(inputs.root), "--video-map", str(inputs.video_map),
         "--out", str(out), "--eligibility", str(inputs.eligibility),
         "--release-manifest", str(inputs.release_manifest), "--workers", "1", *extra],
        capture_output=True, text=True)


def crop_files(out: Path) -> list[Path]:
    return sorted(out.rglob("crop_*"))


def test_crops_come_from_the_mapped_video_with_the_released_roi_and_skip(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36", "72")

    assert result.returncode == 0, result.stderr + result.stdout
    boxes = np.array([BOX] * FRAMES, dtype=float)
    (x1, y1, x2, y2), _ = crop_box(boxes, WIDTH, HEIGHT)
    for size in (36, 72):
        crops = np.load(out / SESSION / f"crop_{size}.npy")
        expected = np.stack([cv2.resize(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB), (size, size),
                                        interpolation=cv2.INTER_AREA) for frame in inputs.frames[SKIP:]])
        assert crops.dtype == np.uint8 and crops.shape == (FRAMES - SKIP, size, size, 3)
        assert np.array_equal(crops, expected)
    assert not (out / SESSION / "crop_128.npy").exists()
    bvp = np.load(out / SESSION / "bvp.npy")
    assert np.allclose(bvp, [np.sin(index / 5) for index in range(SKIP, FRAMES)])
    stats = json.loads((out / SESSION / "stats.json").read_text())
    assert stats["schema_version"] == "compact-v1" and stats["arm"] == "untransformed_control"
    assert stats["frames"] == FRAMES - SKIP and stats["skip_frames"] == SKIP
    assert set(stats["sizes"]) == {"36", "72"}
    assert stats["sizes"]["72"] == session_stats(np.load(out / SESSION / "crop_72.npy"))
    assert stats["cache_provenance"]["video_map_sha256"] == hashlib.sha256(inputs.video_map.read_bytes()).hexdigest()
    check = stats["source_check"]
    assert check["frames"][0] == SKIP and check["frames"][-1] == FRAMES - 1 and len(check["frames"]) == 3
    assert 0 <= check["max_mean_abs_diff"] <= check["limit"] == builder.MAX_MEAN_ABS_DIFF
    with (out / "manifest.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [{"session": SESSION, "participant_key": "demo", "task": "T00",
                     "frames": str(FRAMES - SKIP), "start_frame": str(SKIP), "common_eval_frames": str(FRAMES - SKIP)}]


def test_same_crops_and_statistics_as_the_released_builder_for_the_same_video(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    shutil.copyfile(inputs.root / "demo" / "T00" / "video_bfdt_ffv1.avi", inputs.control)
    released_cache, control_cache = tmp_path / "released_cache", tmp_path / "control_cache"

    reference = subprocess.run(
        [sys.executable, str(RELEASED_BUILDER), "--root", str(inputs.root), "--out", str(released_cache),
         "--eligibility", str(inputs.eligibility), "--release-manifest", str(inputs.release_manifest),
         "--workers", "1"], capture_output=True, text=True)
    result = run(inputs, control_cache, "--verify-frames", "0")

    assert reference.returncode == 0, reference.stderr
    assert result.returncode == 0, result.stderr
    for name in ("crop_36.npy", "crop_72.npy", "crop_128.npy", "bvp.npy"):
        assert (control_cache / SESSION / name).read_bytes() == (released_cache / SESSION / name).read_bytes(), name
    expected = json.loads((released_cache / SESSION / "stats.json").read_text())
    stats = json.loads((control_cache / SESSION / "stats.json").read_text())
    for key in ("schema_version", "session", "participant_key", "task", "frames", "decoded_frames",
                "video_frames_header", "bvp_rows", "skip_frames", "common_eval_frames", "roi", "bbox_rows",
                "valid_bbox_rows", "bbox_policy", "sizes", "label", "moment_precision"):
        assert stats[key] == expected[key], key
    assert stats["source_check"]["frames"] == []
    assert (control_cache / "manifest.csv").read_bytes() == (released_cache / "manifest.csv").read_bytes()


def test_source_paths_are_not_written_to_the_cache_or_the_log(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode == 0, result.stderr
    private = str(tmp_path / "private")
    assert private not in result.stdout and private not in result.stderr
    for path in out.rglob("*"):
        if path.suffix in {".json", ".csv"}:
            assert private not in path.read_text()


@pytest.mark.parametrize("problem", ["missing", "not_a_video"])
def test_failures_do_not_reveal_the_source_path(tmp_path: Path, problem: str) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"
    if problem == "missing":
        inputs.control.unlink()
    else:
        inputs.control.write_text("this is not a video\n")

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode != 0
    assert "cannot be opened" in result.stdout + result.stderr
    private = str(tmp_path / "private")
    assert private not in result.stdout and private not in result.stderr
    assert "source.avi" not in result.stdout and "source.avi" not in result.stderr
    for path in out.rglob("*"):
        if path.suffix in {".json", ".csv"}:
            assert private not in path.read_text()
    assert crop_files(out) == []


def test_frame_count_mismatch_with_the_released_video_is_an_error(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, control_frames=FRAMES - 1)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode != 0
    assert "does not match the released video" in result.stdout + result.stderr
    assert not (out / "manifest.csv").exists()
    assert crop_files(out) == []


def test_frame_size_mismatch_with_the_released_video_is_an_error(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, control_size=(WIDTH + 16, HEIGHT))

    result = run(inputs, tmp_path / "cache", "--sizes", "36")

    assert result.returncode != 0
    assert "does not match the released video" in result.stdout + result.stderr


def test_video_of_another_recording_is_rejected_and_leaves_no_crops(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, control_seed=OTHER_RECORDING)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode != 0
    assert "does not match the released video content" in result.stdout + result.stderr
    assert not (out / "manifest.csv").exists()
    assert not (out / SESSION / "stats.json").exists()
    assert crop_files(out) == []


@pytest.mark.parametrize("source", ["same_file", "copy"])
def test_released_video_is_refused_as_the_source_video(tmp_path: Path, source: str) -> None:
    inputs = make_inputs(tmp_path)
    released_video = inputs.root / "demo" / "T00" / "video_bfdt_ffv1.avi"
    out = tmp_path / "cache"
    if source == "same_file":
        write_video_map(inputs.video_map, [(SESSION, str(released_video))])
    else:
        shutil.copyfile(released_video, inputs.control)

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode != 0
    assert "is the released video itself" in result.stdout + result.stderr
    assert not (out / SESSION / "stats.json").exists()
    assert crop_files(out) == []


def test_content_check_can_be_switched_off(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, control_seed=OTHER_RECORDING)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36", "--verify-frames", "0")

    assert result.returncode == 0, result.stderr
    stats = json.loads((out / SESSION / "stats.json").read_text())
    assert stats["source_check"]["frames"] == []
    assert stats["cache_provenance"]["verify_frames"] == 0


def test_finished_session_is_reused_and_a_changed_source_video_is_refused(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"
    first = run(inputs, out, "--sizes", "36")
    crops_before = (out / SESSION / "crop_36.npy").read_bytes()

    second = run(inputs, out, "--sizes", "36")
    status = os.stat(inputs.control)
    os.utime(inputs.control, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
    third = run(inputs, out, "--sizes", "36")

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert '"status": "reused"' in second.stderr
    assert third.returncode != 0
    assert "built from different inputs" in third.stderr
    assert (out / SESSION / "crop_36.npy").read_bytes() == crops_before
    assert (out / "manifest.csv").exists() is False, "a manifest of an earlier run must not outlive a failed run"


def test_finished_session_with_a_missing_crop_file_is_not_reused(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"
    first = run(inputs, out, "--sizes", "36", "72")
    (out / SESSION / "crop_72.npy").unlink()

    second = run(inputs, out, "--sizes", "36", "72")

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "incomplete" in second.stderr


def test_packed_session_is_reused_only_with_its_index_file(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"
    first = run(inputs, out, "--sizes", "36")
    (out / SESSION / "crop_36.npy").rename(out / SESSION / "crop_36.zlib")
    (out / SESSION / "crop_36.zlib.json").write_text("{}\n")

    with_index = run(inputs, out, "--sizes", "36")
    (out / SESSION / "crop_36.zlib.json").unlink()
    without_index = run(inputs, out, "--sizes", "36")

    assert first.returncode == 0, first.stderr
    assert with_index.returncode == 0, with_index.stderr
    assert '"status": "reused"' in with_index.stderr
    assert without_index.returncode != 0
    assert "incomplete" in without_index.stderr
    assert (out / SESSION / "crop_36.zlib").exists()


def test_session_subset_writes_a_subset_report_and_no_manifest(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36", "--sessions", SESSION)

    assert result.returncode == 0, result.stderr
    assert (out / "build_report_subset.json").exists()
    assert not (out / "manifest.csv").exists()


def test_unknown_session_name_is_rejected_before_anything_is_built(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36", "--sessions", SESSION, "demo_T99")

    assert result.returncode != 0
    assert "unknown or not eligible" in result.stderr
    assert not out.exists()


def test_session_listed_twice_in_the_eligibility_file_is_rejected(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    lines = inputs.eligibility.read_text().splitlines(keepends=True)
    inputs.eligibility.write_text("".join(lines + lines[1:]))
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36")

    assert result.returncode != 0
    assert "more than once" in result.stderr
    assert not out.exists()


def test_negative_number_of_compared_frames_is_rejected(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path, control_seed=OTHER_RECORDING)
    out = tmp_path / "cache"

    result = run(inputs, out, "--sizes", "36", "--verify-frames", "-1")

    assert result.returncode != 0
    assert "must be 0 or more" in result.stderr
    assert not out.exists()


def test_session_without_a_mapped_video_is_an_error(tmp_path: Path) -> None:
    inputs = make_inputs(tmp_path)
    write_video_map(inputs.video_map, [])

    result = run(inputs, tmp_path / "cache", "--sizes", "36")

    assert result.returncode != 0
    assert "no source video" in result.stderr


@pytest.mark.parametrize(
    ("header", "rows", "message"),
    [
        (("session", "video_path"), [(SESSION, "a.avi"), (SESSION, "b.avi")], "session listed twice"),
        (("session", "video_path"), [(SESSION, "")], "empty video_path"),
        (("session", "file"), [(SESSION, "a.avi")], "columns session,video_path"),
    ],
)
def test_unusable_video_map_is_reported_without_a_traceback(
    tmp_path: Path, header: tuple[str, str], rows: list[tuple[str, str]], message: str
) -> None:
    inputs = make_inputs(tmp_path)
    write_video_map(inputs.video_map, rows, header)

    result = run(inputs, tmp_path / "cache", "--sizes", "36")

    assert result.returncode != 0
    assert message in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("through_symlink", [False, True])
def test_cache_inside_the_repository_is_refused(tmp_path: Path, through_symlink: bool) -> None:
    inputs = make_inputs(tmp_path)
    inside = REPO / "benchmark" / "_control_cache_must_not_be_created"
    out = inside
    if through_symlink:
        out = tmp_path / "link_into_repository"
        out.symlink_to(inside)

    try:
        result = run(inputs, out, "--sizes", "36")

        assert result.returncode != 0
        assert "outside the repository" in result.stderr
        assert not inside.exists()
    finally:
        if inside.exists():
            for leftover in sorted(inside.rglob("*"), reverse=True):
                leftover.unlink() if leftover.is_file() else leftover.rmdir()
            inside.rmdir()


def job_for(inputs: Inputs, out: Path) -> tuple:
    with inputs.eligibility.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    return (str(inputs.root), str(out), row, str(inputs.control), (36,), {"built_by": "test"}, 3)


def test_source_video_that_changes_while_it_is_read_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"
    read_crops = builder.read_crops

    def read_then_touch(video_path: str, *rest: object) -> tuple:
        result = read_crops(video_path, *rest)
        status = os.stat(video_path)
        os.utime(video_path, ns=(status.st_atime_ns, status.st_mtime_ns + 10**9))
        return result

    monkeypatch.setattr(builder, "read_crops", read_then_touch)

    with pytest.raises(ValueError, match="changed while it was read"):
        builder.build_one(job_for(inputs, out))

    assert not (out / SESSION / "stats.json").exists()
    assert crop_files(out) == []


def test_unexpected_error_text_does_not_show_the_source_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = make_inputs(tmp_path)
    out = tmp_path / "cache"

    def fail(video_path: str, *rest: object) -> tuple:
        raise RuntimeError(f"cannot read {video_path}")

    monkeypatch.setattr(builder, "read_crops", fail)
    monkeypatch.setattr(builder, "ProcessPoolExecutor", ThreadPoolExecutor)  # the worker must see the patched reader
    previous_umask = os.umask(0o077)
    try:
        code = builder.main(
            ["--root", str(inputs.root), "--video-map", str(inputs.video_map), "--out", str(out),
             "--eligibility", str(inputs.eligibility), "--release-manifest", str(inputs.release_manifest),
             "--workers", "1", "--sizes", "36"])
    finally:
        os.umask(previous_umask)

    report = (out / "build_report_all.json").read_text()
    assert code == 1
    assert "cannot read <source video>" in report
    assert str(tmp_path / "private") not in report
    assert crop_files(out) == []


@pytest.mark.parametrize("is_finished", [False, True])
def test_only_an_unfinished_session_is_discarded(tmp_path: Path, is_finished: bool) -> None:
    dest = tmp_path / SESSION
    dest.mkdir()
    names = ["crop_36.npy", "crop_72.npy", "bvp.npy", "stats.json.tmp"] + (["stats.json"] if is_finished else [])
    for name in names:
        (dest / name).write_text("x")

    builder.discard_partial(dest)

    assert sorted(path.name for path in dest.iterdir()) == (sorted(names) if is_finished else [])
