#!/usr/bin/env python3
"""Turn rPPG-Toolbox test-output pickles into one per-session heart-rate table.

    python benchmark/collect_predictions.py kr_kfold_eval      # -> $KR_RPPG_WORK/results/kr_kfold_eval.csv
    python benchmark/collect_predictions.py kr_to_pure pure_to_kr mmpd_to_kr_eval

Arguments are run families, i.e. folders under $KR_RPPG_WORK/runs/ (LOG.PATH of the configs). Every
`saved_test_outputs/*_outputs.pickle` below them is scored with the toolbox post-processing
(evaluation.post_process.calculate_metric_per_video: cumsum for DiffNormalized labels, smoothness-priors
detrend lambda=100, 1st-order Butterworth 0.6-3.3 Hz, periodogram peak over the whole session), i.e. the
same numbers the toolbox prints as "FFT MAE (FFT Label)".

Output columns: family, model, fold, session, participant_key, task, pred_hr, label_hr, abs_error.
The table is participant-level for KR-rPPG: it is written under $KR_RPPG_WORK (keep it on access-controlled storage),
never inside this repository. Only load pickles you produced yourself (pickle can execute code).
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List

from common import MODELS, REPO, import_toolbox, work_dir

RUN_DIR = re.compile(r"^(?P<model>" + "|".join(MODELS) + r")(?:_fold(?P<fold>\d))?$")
KR_SESSION = re.compile(r"^(?P<pid>p\d{3})_(?P<task>T0[0-3])$")
FIELDS = ["family", "model", "fold", "session", "participant_key", "task", "pred_hr", "label_hr", "abs_error"]


def score_pickle(path: Path) -> Iterator[Dict]:
    from evaluation.metrics import _reform_data_from_dict
    from evaluation.post_process import calculate_metric_per_video

    with open(path, "rb") as handle:
        data = pickle.load(handle)  # own toolbox output only
    diff_flag = data.get("label_type", "DiffNormalized") == "DiffNormalized"
    fs = data.get("fs", 30)
    for session in sorted(data["predictions"]):
        pred = _reform_data_from_dict(data["predictions"][session])
        label = _reform_data_from_dict(data["labels"][session])
        if len(pred) < 9:
            continue  # the toolbox ignores windows shorter than 9 samples
        label_hr, pred_hr, _snr, _macc = calculate_metric_per_video(
            pred, label, fs=fs, diff_flag=diff_flag, use_bandpass=True, hr_method="FFT")
        yield {"session": str(session), "pred_hr": float(pred_hr), "label_hr": float(label_hr)}


def run_info(pickle_path: Path, family_dir: Path) -> Dict[str, str]:
    """model and fold from the run folder name (<model> or <model>_fold<k>) directly below the family."""
    rel = pickle_path.relative_to(family_dir).parts
    match = RUN_DIR.match(rel[0])
    if not match:
        raise ValueError(f"cannot tell model/fold from run folder {rel[0]!r}")
    return {"model": match["model"], "fold": match["fold"] or ""}


def collect(family: str, runs: Path) -> List[Dict]:
    family_dir = runs / family
    pickles = sorted(family_dir.rglob("saved_test_outputs/*_outputs.pickle"))
    if not pickles:
        sys.exit(f"no test outputs below {family_dir}")
    rows = []
    for path in pickles:
        info = run_info(path, family_dir)
        for row in score_pickle(path):
            match = KR_SESSION.match(row["session"])
            rows.append({
                "family": family, **info, **row,
                "participant_key": match["pid"] if match else "",
                "task": match["task"] if match else "",
                "abs_error": abs(row["pred_hr"] - row["label_hr"]),
            })
        print(f"[collect] {family}: {path.relative_to(family_dir).parts[0]} -> {len(rows)} rows so far", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("families", nargs="+")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: $KR_RPPG_WORK/results")
    args = parser.parse_args()
    work = work_dir()
    out_dir = (args.out_dir or work / "results").resolve()
    if out_dir == REPO or REPO in out_dir.parents:
        sys.exit("refusing to write per-session results inside the repository")
    import_toolbox()
    out_dir.mkdir(parents=True, exist_ok=True)
    for family in args.families:
        rows = collect(family, work / "runs")
        out = out_dir / f"{family}.csv"
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[collect] wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
