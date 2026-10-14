#!/usr/bin/env python3
"""Figure 3 values: task-stratified session-level MAE of the five models (KR-rPPG 5-fold, out-of-fold).

    python analysis/figure3/figure3_values.py                               # $KR_RPPG_WORK/results/kr_kfold_eval.csv
    python analysis/figure3/figure3_values.py --predictions my.csv --out values.json
    python analysis/figure3/figure3_values.py --compare analysis/figure3/expected_values.json

Method (manuscript Figure 3 caption):
  * reference HR = label HR shared by the three 180-frame models (DeepPhys, TS-CAN, EfficientPhys),
    so all five models are scored against the same contact-pulse reference;
  * session error = |predicted HR - reference HR| (bpm);
  * cell MAE = mean within participant, then mean across participants (at most one session per
    participant and task, so per task this equals the session mean);
  * 95% CI = percentile bootstrap over participants (sorted by participant_key), B = 10,000,
    numpy.random.default_rng(42) created afresh for every cell, rng.integers(0, n, size=(B, n)).
The "All" group is the Table 8 internal row. Output: aggregates only (no participant-level values).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MODELS = {"physnet": "PhysNet", "physformer": "PhysFormer", "deepphys": "DeepPhys", "tscan": "TS-CAN",
          "efficientphys": "EfficientPhys"}
REFERENCE_MODELS = ("deepphys", "tscan", "efficientphys")
TASKS = ("T00", "T01", "T02", "T03")
GROUPS = TASKS + ("All",)
B, SEED, LARGE_ERROR_BPM = 10_000, 42, 5.0


def common_reference(df: pd.DataFrame) -> pd.Series:
    labels = df[df.model.isin(REFERENCE_MODELS)].pivot_table(
        index="session", columns="model", values="label_hr", aggfunc="first")
    spread = (labels.max(axis=1) - labels.min(axis=1)).max()
    if not np.isfinite(spread) or spread > 1e-9:
        print(f"warning: the 180-frame models disagree on the label HR (max spread {spread:.3g} bpm); "
              "using DeepPhys", file=sys.stderr)
    return labels["deepphys"].rename("ref_hr")


def bootstrap_ci(values: np.ndarray) -> Tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(values), size=(B, len(values)))
    boot = values[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cell(sub: pd.DataFrame) -> Dict:
    ae = sub["ae"].to_numpy(dtype=float)
    keys = sub["participant_key"].to_numpy()
    uniq, inv = np.unique(keys, return_inverse=True)  # sorted participant keys
    per_participant = np.bincount(inv, weights=ae) / np.bincount(inv)
    lo, hi = bootstrap_ci(per_participant)
    return {"n_sessions": int(len(ae)), "n_participants": int(len(uniq)), "mae": float(per_participant.mean()),
            "ci95_lo": lo, "ci95_hi": hi, "median_ae": float(np.median(ae)),
            "pct_sessions_gt5": float(100 * (ae > LARGE_ERROR_BPM).mean())}


def compute(df: pd.DataFrame) -> Dict:
    df = df.merge(common_reference(df), left_on="session", right_index=True, how="left")
    if df["ref_hr"].isna().any():
        sys.exit(f"{int(df['ref_hr'].isna().sum())} rows have no common reference HR")
    df = df.assign(ae=(df["pred_hr"] - df["ref_hr"]).abs())
    out = {}
    for model, name in MODELS.items():
        dm = df[df.model == model]
        if dm.empty:
            continue
        out[name] = {g: cell(dm if g == "All" else dm[dm.task == g]) for g in GROUPS}
    return out


def compare(values: Dict, expected_path: Path) -> float:
    expected = json.loads(expected_path.read_text())["values"]
    worst = 0.0
    print(f"{'model':14s} {'group':5s} {'MAE':>7s} {'expected':>9s}")
    for model, groups in expected.items():
        for group, exp in groups.items():
            got = values.get(model, {}).get(group)
            if got is None:
                continue
            worst = max(worst, abs(round(got["mae"], 2) - exp["mae"]))
            print(f"{model:14s} {group:5s} {got['mae']:7.2f} {exp['mae']:9.2f}")
    print(f"max |difference| at 2 dp: {worst:.2f} bpm")
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="collect_predictions.py output (default: $KR_RPPG_WORK/results/kr_kfold_eval.csv)")
    parser.add_argument("--out", type=Path, default=None, help="default: <predictions folder>/figure3_values.json")
    parser.add_argument("--compare", type=Path, default=None, help="expected values JSON to compare with")
    args = parser.parse_args()
    if args.predictions is None:
        if not os.environ.get("KR_RPPG_WORK"):
            sys.exit("give --predictions or set KR_RPPG_WORK")
        args.predictions = Path(os.environ["KR_RPPG_WORK"]) / "results" / "kr_kfold_eval.csv"
    df = pd.read_csv(args.predictions, dtype={"participant_key": str, "session": str, "fold": str})
    values = compute(df)
    out = args.out or args.predictions.with_name("figure3_values.json")
    meta = {"method": "common 180-frame reference HR; participant mean; percentile bootstrap B=10000 seed 42",
            "source": args.predictions.name}
    out.write_text(json.dumps({"meta": meta, "values": values}, indent=1))
    for model, groups in values.items():
        print(f"{model:14s} " + " ".join(f"{g}:{v['mae']:.2f}[{v['ci95_lo']:.2f},{v['ci95_hi']:.2f}]"
                                          for g, v in groups.items()))
    print(f"wrote {out}")
    if args.compare:
        compare(values, args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
