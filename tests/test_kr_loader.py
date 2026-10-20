"""KR-rPPG loader on a tiny synthetic release tree (pNNN/T0x layout, BFDT-style FFV1 video, metadata/)."""

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "third_party" / "rPPG-Toolbox"))

from dataset.data_loader.KoreanrPPGLoader import KoreanrPPGLoader  # noqa: E402

PARTICIPANTS = ("p901", "p902", "p903", "p904", "p905", "p906", "p907", "p908", "p909", "p910")
N_FRAMES, W, H = 330, 64, 48  # 6 s are skipped, leaving 150 frames = one 128-frame chunk


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _session(folder: Path, seed: int):
    folder.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(folder / "video_bfdt_ffv1.avi"), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (W, H), True)
    assert writer.isOpened()
    for _ in range(N_FRAMES):
        writer.write(rng.integers(0, 256, (H, W, 3), dtype=np.uint8))
    writer.release()
    t = np.arange(N_FRAMES) / 30.0
    _write_csv(folder / "BVP_synchronized.csv", ["timestamp", "bvp"],
               [[f"{x:.6f}", f"{np.sin(2 * np.pi * 1.2 * x):.6f}"] for x in t])
    _write_csv(folder / "face_bboxes.csv", ["x1", "y1", "x2", "y2"], [[8, 8, 40, 40]] * N_FRAMES)


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "KR-rPPG"
    for i, pid in enumerate(PARTICIPANTS):
        tasks = ("T00", "T01") if i % 2 else ("T01",)
        for j, task in enumerate(tasks):
            _session(root / pid / task, seed=10 * i + j)
    (root / "metadata").mkdir()
    _write_csv(root / "metadata" / "fold_assignment.csv", ["participant_key", "fold"],
               [[pid, i % 5] for i, pid in enumerate(PARTICIPANTS)])
    (root / "p911").mkdir()  # participant folder without sessions is ignored
    return root


def _loader(root, name="train"):
    loader = KoreanrPPGLoader.__new__(KoreanrPPGLoader)  # skip BaseLoader.__init__ (no cache needed)
    loader.dataset_name = name
    loader.inputs, loader.labels = [], []
    loader.config_data = SimpleNamespace(FS=30, KFOLD_CSV=str(root / "metadata" / "fold_assignment.csv"),
                                         KFOLD_ID=1, KFOLD_TEST_FOLD=False, KFOLD_SPLIT_CSV="")
    return loader


def test_sessions_are_indexed_by_release_id(release):
    dirs = _loader(release).get_raw_data(str(release))
    assert len(dirs) == 15
    assert dirs[0]["index"] == "p901_T01" and dirs[1]["index"] == "p902_T00"
    assert {d["participant"] for d in dirs} == set(PARTICIPANTS)


def test_kfold_test_fold_and_train_pool(release):
    dirs = _loader(release).get_raw_data(str(release))
    test_loader = _loader(release, "test")
    test_loader.config_data.KFOLD_TEST_FOLD = True
    assert {d["participant"] for d in test_loader.split_raw_data(dirs, 0.0, 1.0)} == {"p902", "p907"}
    train = {d["participant"] for d in _loader(release).split_raw_data(dirs, 0.0, 0.9)}
    valid = {d["participant"] for d in _loader(release, "valid").split_raw_data(dirs, 0.9, 1.0)}
    assert train.isdisjoint(valid) and len(train) == 7 and valid == {"p910"}  # last 10% of the sorted pool
    assert not ({"p902", "p907"} & (train | valid))


def test_split_csv_overrides_the_ratio_split(release, tmp_path):
    split = tmp_path / "split.csv"
    _write_csv(split, ["fold", "participant_key", "split"],
               [[1, "p901", "valid"], [1, "p903", "train"], [1, "p904", "train"], [0, "p901", "train"]])
    dirs = _loader(release).get_raw_data(str(release))
    train_loader, valid_loader = _loader(release, "train"), _loader(release, "valid")
    for loader in (train_loader, valid_loader):
        loader.config_data.KFOLD_SPLIT_CSV = str(split)
    assert {d["participant"] for d in train_loader.split_raw_data(dirs, 0.0, 0.9)} == {"p903", "p904"}
    assert {d["participant"] for d in valid_loader.split_raw_data(dirs, 0.9, 1.0)} == {"p901"}


def test_preprocessing_one_session_writes_chunks(release, tmp_path):
    loader = _loader(release, "test")
    loader.cached_path = str(tmp_path / "cache")
    Path(loader.cached_path).mkdir()
    pre = SimpleNamespace(
        DATA_AUG=["None"], USE_PSUEDO_PPG_LABEL=False, DATA_TYPE=["DiffNormalized", "Standardized"],
        LABEL_TYPE="DiffNormalized", DO_CHUNK=True, CHUNK_LENGTH=128,
        CROP_FACE=SimpleNamespace(USE_LARGE_FACE_BOX=True, LARGE_BOX_COEF=1.5), RESIZE=SimpleNamespace(W=72, H=72))
    dirs = loader.get_raw_data(str(release))
    result = {}
    loader.preprocess_dataset_subprocess(dirs, pre, 0, result)
    assert len(result[0]) == 1
    x = np.load(result[0][0])
    y = np.load(result[0][0].replace("input", "label"))
    assert x.shape == (128, 72, 72, 6) and y.shape == (128,)
    assert np.isfinite(x).all() and np.isfinite(y).all()
