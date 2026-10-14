"""Synthetic <source_root> tree shared by the tests.

Layout mirrors the real dataset: <site>/<group>/pNNN/T0x/ with an FFV1 video, the
synchronized CSVs (epoch timestamps, sync files off by ~2.4e-7 s after a float64 round trip),
face boxes, info.txt with a recording date, and a metadata/ folder keyed by `site_pNNN`.

Every value below is invented. Nothing is copied from a recording or from a case report form:
the session clock starts at the round epoch 1,000,000,000 s (2001-09-09T01:46:40Z), the info.txt
date is the placeholder 2000-01-01, signal samples follow simple formulas, and the CRF rows are
drawn by a seeded generator (see `synthetic_crf_rows`). Participant numbers use the reserved 9NN range.
"""

import csv
import hashlib
import os
import random
import stat
from pathlib import Path

import cv2
import numpy as np
import pytest

from krrppg_release.clinical import CRF_FILENAME

FRAME_W, FRAME_H = 64, 48
N_FRAMES = 6
EPOCH_T0 = "1000000000.0000000"  # synthetic: round epoch, outside any recording period
INFO_DATE = "2000-01-01 00:00:00"  # synthetic placeholder date
FRAME_STEP_S = 0.0333  # synthetic inter-frame interval
FLOAT_ROUND_TRIP_S = 2.384185791015625e-07  # 2**-22 s: the size of a float64 round-trip difference

# participant_key -> (site, group, pid, {task: released_flag})
PARTICIPANTS = {
    "incheon_p901": ("incheon", "02-groupb", "p901", {"T00": 1, "T01": 1}),
    "incheon_p902": ("incheon", "01-groupa", "p902", {"T00": 1, "T01": 0}),
    "seoul_p903": ("seoul", "01-groupa", "p903", {"T02": 1}),
}
UNRELEASED_DIR = ("seoul", "01-groupa", "p907")


def _write_csv(path, header, rows, newline="\n"):
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator=newline)
        writer.writerow(header)
        writer.writerows(rows)


def _timestamps(n):
    base = float(EPOCH_T0)
    return [repr(base + i * FRAME_STEP_S) if i else EPOCH_T0 for i in range(n)]


def _ulp_shift(value):
    return repr(float(value) + FLOAT_ROUND_TRIP_S)


def make_video(path, frames):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), 30.0, (FRAME_W, FRAME_H), True)
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def make_session(task_dir, seed, n_frames=N_FRAMES):
    task_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    frames = [rng.integers(0, 256, (FRAME_H, FRAME_W, 3), dtype=np.uint8) for _ in range(n_frames)]
    make_video(task_dir / "video_ZIP_RAW_BGRA.avi", frames)
    ts = _timestamps(n_frames)
    _write_csv(task_dir / "frames_timestamp.csv", ["frame", "timestamp"], [[i, t] for i, t in enumerate(ts)], "\r\n")
    sync_ts = [_ulp_shift(t) if i % 2 else t for i, t in enumerate(ts)]
    _write_csv(task_dir / "BVP_synchronized.csv", ["timestamp", "bvp"], [[t, f"{i * 0.5 - 1.25}"] for i, t in enumerate(sync_ts)])
    _write_csv(task_dir / "HR_synchronized.csv", ["frame", "timestamp", "hr"], [[i, t, f"{70.5 + i}"] for i, t in enumerate(sync_ts)])
    _write_csv(task_dir / "SpO2_synchronized.csv", ["frame", "timestamp", "spo2"], [[i, t, "99.5"] for i, t in enumerate(sync_ts)])
    _write_csv(task_dir / "face_bboxes.csv", ["x1", "y1", "x2", "y2"], [[8, 8, 40, 40] for _ in range(n_frames)])
    (task_dir / "info.txt").write_text(f"cam model: Brio 500\ncam codec: YUY2\nsize: 64x48\ndate: {INFO_DATE}")
    # files that must never be released
    (task_dir / "BVP.csv").write_text(f"timestamp,bvp\n{EPOCH_T0},-1.25\n")
    (task_dir / "face_bboxes_G.csv").write_text("x1,y1,x2,y2\n1,2,3,4\n")
    return frames


def make_metadata(meta_dir):
    meta_dir.mkdir(parents=True)
    keys = sorted(PARTICIPANTS)
    _write_csv(
        meta_dir / "participants.csv",
        ["participant_key", "site", "age", "sex", "height"],
        [[k, PARTICIPANTS[k][0], 60 + i, "M", "170.0"] for i, k in enumerate(keys)],
    )
    _write_csv(
        meta_dir / "participants_dictionary.csv",
        ["variable", "variable_group", "description", "unit_or_coding", "availability_n_of_296"],
        [
            ["participant_key", "Demographics", "Pseudonymized participant identifier ({site}_pNNN; e.g., incheon_p902)", "-", "3"],
            ["site", "Demographics", "Collection site", "-", "3"],
            ["age", "Demographics", "Age at enrollment", "years", "3"],
            ["sex", "Demographics", "Biological sex", "M/F", "3"],
            ["height", "Demographics", "Body height", "cm, rounded to 5", "3"],
        ],
    )
    _write_csv(
        meta_dir / "task_session_availability.csv",
        ["participant_key", "t00_released", "t01_released", "t02_released", "t03_released"],
        [[k] + [PARTICIPANTS[k][3].get(t, 0) for t in ("T00", "T01", "T02", "T03")] for k in keys],
    )
    _write_csv(meta_dir / "fold_assignment.csv", ["participant_key", "fold"], [[k, i % 5] for i, k in enumerate(keys)])
    (meta_dir / "README.md").write_text("internal notes - not released\n")


# --- synthetic CRF ----------------------------------------------------------------------------

CRF_COLUMNS = [
    "Subject Screening NO", "Age", "Sex", "Height", "Weight", "SBP", "DBP", "HR", "HbA1c", "FPG",
    "Creatinine", "GFR(CKD-EPI)", "AST", "ALT", "Makeup", "Glasses", "Accessories", "DM", "HTN",
    "Dyslipidemia", "Cardiovascular", "Cerebrovascular", "Cancer", "Renal disease", "Lung disease",
]
CRF_SEED = 2000  # the rows are a pure function of this seed (random.Random is stable across Python versions)
PRESENT, ABSENT = 1, 2  # CRF coding of binary items; '-' and 'ND' = missing
MALE, FEMALE = 1, 2


def _flags(makeup, glasses, accessories, dm, htn, dyslipidemia, cardio, cerebro, cancer, renal, lung):
    names = CRF_COLUMNS[14:]
    values = (makeup, glasses, accessories, dm, htn, dyslipidemia, cardio, cerebro, cancer, renal, lung)
    return {name: PRESENT if value else ABSENT for name, value in zip(names, values)}


# One spec per row. (lo, hi) = draw uniformly in that range; ("years", lo, hi) = the same draw rounded to whole
# years (ages are whole numbers in the CRF); ("ckd-epi", lo, hi) = CKD-EPI 2009 eGFR of the drawn creatinine/age/sex
# plus an offset drawn in [lo, hi]; ("above", column, lo, hi) = that column's drawn value plus an offset drawn in
# [lo, hi]; anything else is written as is. Every draw consumes one random number, and each row trips exactly the
# data-quality checks named in its comment (since 1.4.0 the checks only count; no value is removed).
CRF_SPECS = [
    # incheon_p901: everything recorded; CRF eGFR within 3 of CKD-EPI (kept); other comorbidities all absent
    {"Subject Screening NO": "04-S901", "Age": ("years", 50.5, 59.5), "Sex": FEMALE, "Height": (162, 166),
     "Weight": (71, 76), "SBP": (131, 138), "DBP": (81, 88), "HR": (62, 95), "HbA1c": (5.2, 6.3),
     "FPG": (101, 124), "Creatinine": (0.80, 0.90), "GFR(CKD-EPI)": ("ckd-epi", -3, 3), "AST": (15, 38),
     "ALT": (41.5, 60), **_flags(1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0)},
    # incheon_p902: CRF eGFR missing ('-') -> computed from creatinine; FPG 'ND'; BMI < 18.5 (not a check);
    # cerebrovascular disease and cancer present (count 2)
    {"Subject Screening NO": "04-S902", "Age": ("years", 74, 78.5), "Sex": MALE, "Height": (174, 178),
     "Weight": (51, 55), "SBP": (161, 175), "DBP": (91, 98), "HR": (101, 115), "HbA1c": (8.2, 9.4),
     "FPG": "ND", "Creatinine": (1.20, 1.35), "GFR(CKD-EPI)": "-", "AST": (42, 55), "ALT": (18, 35),
     **_flags(1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0)},
    # seoul_p903: height > 210 cm and CRF BMI < 13 (both counted, values kept); DBP above SBP (counted, kept);
    # CRF eGFR < 15 and consistent with creatinine (counted as renal disease); ILD not ascertained outside incheon
    {"Subject Screening NO": "01-S903", "Age": ("years", 80.5, 88), "Sex": FEMALE, "Height": (212.5, 225),
     "Weight": (50, 58), "SBP": (100, 115), "DBP": ("above", "SBP", 1, 12), "HR": (50, 58), "HbA1c": (6.55, 6.85),
     "FPG": (130, 160), "Creatinine": (3.6, 4.8), "GFR(CKD-EPI)": ("ckd-epi", -1, 1), "AST": (20, 35),
     "ALT": (12, 30), **_flags(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0)},
    # seoul_p907: in the CRF but never released
    {"Subject Screening NO": "01-S907", "Age": ("years", 61, 68), "Sex": MALE, "Height": (166, 174),
     "Weight": (63, 69), "SBP": (121, 128), "DBP": (72, 79), "HR": (65, 85), "HbA1c": (5.5, 6.2),
     "FPG": (90, 99), "Creatinine": (0.95, 1.10), "GFR(CKD-EPI)": ("ckd-epi", -4, 4), "AST": (18, 30),
     "ALT": (15, 28), **_flags(0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1)},
]
NOT_A_SCREENING_ROW = ["99-X001"] + [0] * (len(CRF_COLUMNS) - 1)  # not a screening number: dropped by the loader
ILD_ROWS = [["04-S901", "X"], ["04-S902", "O"], ["01-S903", "-"], ["01-S907", "-"]]


def _draw(rng, lo, hi):
    """Uniform in [lo, hi] with exactly three decimals (never a whole number of hundredths)."""
    value = round(lo + (hi - lo) * rng.random(), 3)
    if round(value, 2) == value:
        value = round(value + 0.001, 3)
    return value


def ckd_epi_2009(creatinine, age, female):
    """CKD-EPI 2009 creatinine equation (no race coefficient); independent of the package under test."""
    kappa, alpha = (0.7, -0.329) if female else (0.9, -0.411)
    ratio = creatinine / kappa
    egfr = 141 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.209 * 0.993 ** age
    return egfr * 1.018 if female else egfr


def synthetic_crf_rows(seed=CRF_SEED):
    rng = random.Random(seed)
    rows = []
    for spec in CRF_SPECS:
        row = {}
        for column in CRF_COLUMNS:
            rule = spec[column]
            if isinstance(rule, tuple) and rule[0] == "ckd-epi":
                exact = ckd_epi_2009(row["Creatinine"], row["Age"], row["Sex"] == FEMALE)
                row[column] = round(exact + _draw(rng, rule[1], rule[2]), 3)
            elif isinstance(rule, tuple) and rule[0] == "years":
                row[column] = round(_draw(rng, rule[1], rule[2]))  # round() without digits returns an int
            elif isinstance(rule, tuple) and rule[0] == "above":
                row[column] = round(row[rule[1]] + _draw(rng, rule[2], rule[3]), 3)
            elif isinstance(rule, tuple):
                row[column] = _draw(rng, *rule)
            else:
                row[column] = rule
        rows.append([row[column] for column in CRF_COLUMNS])
    return rows + [NOT_A_SCREENING_ROW]


CRF_ROWS = synthetic_crf_rows()


def _write_sheet(path, sheet, header, rows):
    import pandas as pd

    frame = pd.DataFrame(rows, columns=header)
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name=sheet, startrow=5, index=False)


CRF_SHEET, ILD_SHEET = "crf_synthetic", "ild_synthetic"  # without KRRPPG_*_SHEET the builder reads the first sheet


def make_crf(crf_path, ild_path):
    _write_sheet(crf_path, CRF_SHEET, CRF_COLUMNS, CRF_ROWS)
    _write_sheet(ild_path, ILD_SHEET, ["Subject Screening NO", "ILD"], ILD_ROWS)


def tree_fingerprint(root):
    """sha256 + size + mtime_ns + ctime_ns of every file below root."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            path = Path(dirpath) / name
            st = path.stat()
            out[str(path.relative_to(root))] = (
                hashlib.sha256(path.read_bytes()).hexdigest(),
                st.st_size,
                st.st_mtime_ns,
                st.st_ctime_ns,
            )
    return out


# --- filesystem helpers for the permission tests ------------------------------------------------

NETWORK_FILESYSTEMS = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "fuse.sshfs", "9p", "ceph", "lustre", "gpfs",
                       "glusterfs", "fuse.glusterfs"}


def filesystem_type(path):
    """Type of the filesystem that holds `path`, from /proc/self/mounts ('' when unknown)."""
    real = os.path.realpath(path)
    best, fstype = "", ""
    try:
        with open("/proc/self/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount = parts[1].replace("\\040", " ")
                inside = real == mount or real.startswith(mount.rstrip("/") + "/")
                if inside and len(mount) >= len(best):
                    best, fstype = mount, parts[2]
    except OSError:
        return ""
    return fstype


def on_network_filesystem(path):
    return filesystem_type(path) in NETWORK_FILESYSTEMS


def chmod_or_skip(path, mode):
    """Give a test directory `mode` (which may include the setgid bit), or skip the test if that is impossible.

    chmod silently drops S_ISGID unless the directory's group is one of the caller's groups, and on some NFS
    servers even then; a directory created under a setgid parent (common for shared temp folders) inherits the
    parent's group. Re-owning it to the caller's effective group first makes the bit stick on local disks and NFS.
    """
    try:
        os.chown(path, -1, os.getegid())
    except OSError:
        pass
    os.chmod(path, mode)
    actual = stat.S_IMODE(os.stat(path).st_mode)
    if actual != mode:
        pytest.skip(f"this filesystem keeps mode {oct(actual)} after chmod {oct(mode)}; cannot set up the test")


@pytest.fixture
def fake_source(tmp_path):
    root = tmp_path / "source_root"
    frames = {}
    for i, (key, (site, group, pid, tasks)) in enumerate(sorted(PARTICIPANTS.items())):
        for j, task in enumerate(sorted(tasks)):
            frames[(key, task)] = make_session(root / site / group / pid / task, seed=10 * i + j)
    site, group, pid = UNRELEASED_DIR
    make_session(root / site / group / pid / "T00", seed=99)
    make_metadata(root / "metadata")
    ild_xlsx = tmp_path / "ild_items.xlsx"  # lives outside the source tree, like the real one
    make_crf(root / CRF_FILENAME, ild_xlsx)
    return {"root": root, "frames": frames, "workdir": tmp_path, "ild_xlsx": ild_xlsx}
