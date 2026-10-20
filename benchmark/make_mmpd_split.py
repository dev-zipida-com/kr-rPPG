#!/usr/bin/env python3
"""Write the MMPD subject split used for the MMPD -> KR-rPPG row of Table 8.

    python benchmark/make_mmpd_split.py            # after: python benchmark/preprocess.py preprocess/mmpd_*.yaml

Subjects 1-26 (all recording conditions) are used for training and subjects 27-33 for validation
(checkpoint = lowest validation loss over 10 epochs). For every MMPD cache under
$KR_RPPG_WORK/cache/mmpd/<cache>/ this reads the full file list written by the toolbox
(<EXP_DATA_NAME>_0.0_1.0.csv, column input_files) and writes
$KR_RPPG_WORK/splits/mmpd_<cache>_train_subjects_01_26.csv and ..._valid_subjects_27_33.csv.
MMPD file names start with "subject<N>_", the public MMPD subject number.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

from common import MODELS, work_dir

SUBJECT = re.compile(r"^subject(\d+)_")
SPLITS = {"train_subjects_01_26": range(1, 27), "valid_subjects_27_33": range(27, 34)}


def subject_of(path: str) -> int:
    match = SUBJECT.match(Path(path).name)
    if not match:
        raise ValueError(f"not an MMPD cache file: {Path(path).name}")
    return int(match.group(1))


def main() -> int:
    work = work_dir()
    out_dir = work / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    for cache in MODELS:
        lists = sorted((work / "cache" / "mmpd" / cache / "DataFileLists").glob("MMPD_*_0.0_1.0.csv"))
        if not lists:
            print(f"[mmpd-split] {cache}: no full file list yet (run the MMPD preprocessing first)")
            continue
        if len(lists) > 1:
            sys.exit(f"[mmpd-split] {cache}: several full file lists, keep one: {[p.name for p in lists]}")
        files = pd.read_csv(lists[0])["input_files"].tolist()
        subjects = [subject_of(f) for f in files]
        for name, wanted in SPLITS.items():
            chosen = [f for f, s in zip(files, subjects) if s in wanted]
            pd.DataFrame({"input_files": chosen}).to_csv(out_dir / f"mmpd_{cache}_{name}.csv")
            n_subjects = len({s for s in subjects if s in wanted})
            print(f"[mmpd-split] {cache} {name}: {len(chosen)} clips from {n_subjects} subjects")
        done += 1
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
