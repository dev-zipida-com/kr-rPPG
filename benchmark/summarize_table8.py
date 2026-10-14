#!/usr/bin/env python3
"""Print manuscript Table 8 (session-level MAE, bpm) from the tables of collect_predictions.py.

    python benchmark/summarize_table8.py            # reads $KR_RPPG_WORK/results/*.csv

Row definitions (as in the manuscript):
  * Internal KR-rPPG 5-fold (also Figure 3): every out-of-fold session is scored against ONE common
    reference HR, the label HR of the 180-frame models (DeepPhys, TS-CAN, EfficientPhys share it);
    absolute errors are averaged within participant, then across participants.
  * KR-rPPG -> external: each of the 5 fold models is scored on the whole external dataset with its own
    label HR; the row value is the mean of the 5 fold MAEs.
  * external -> KR-rPPG: one model per architecture scored on every released KR-rPPG session.
Only aggregates are printed. Expected values are those of the manuscript (see README for caveats); rows
without listed reference values print "-" in parentheses.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from common import DISPLAY, MODELS, work_dir

# family (results/<family>.csv), row label, expected manuscript values in MODELS order (None: none listed).
# The external -> KR-rPPG values are MAEs over the 1,166 released sessions, the sessions session_row() averages
# over on the release. Every PhysFormer value is scored with the gra_sharp the model was trained with (README).
ROWS: List[Tuple[str, str, Optional[Tuple[float, ...]]]] = [
    ("kr_kfold_eval", "Internal  KR-rPPG -> KR-rPPG", (1.47, 1.51, 11.83, 7.48, 4.81)),
    ("kr_to_pure", "External  KR-rPPG -> PURE", (3.65, 4.34, 2.74, 0.31, 2.80)),
    ("kr_to_ubfc_rppg", "External  KR-rPPG -> UBFC-rPPG", (1.64, 1.70, 1.85, 2.00, 2.78)),
    ("kr_to_mmpd", "External  KR-rPPG -> MMPD", (8.97, 8.38, 21.45, 15.53, 15.54)),
    ("kr_to_ubfc_phys", "External  KR-rPPG -> UBFC-Phys", (5.11, 5.66, 8.50, 5.79, 6.36)),
    ("pure_to_kr", "External  PURE -> KR-rPPG", (10.83, 10.57, 16.98, 11.51, 10.87)),
    ("ubfc_rppg_to_kr", "External  UBFC-rPPG -> KR-rPPG", (5.19, 7.41, 15.96, 11.65, 11.56)),
    ("mmpd_to_kr_eval", "External  MMPD -> KR-rPPG", None),
]
REFERENCE_MODEL = "deepphys"  # its label HR is the common 180-frame reference (identical for TS-CAN, EfficientPhys)


def internal_row(df: pd.DataFrame) -> Dict[str, float]:
    ref = df[df.model == REFERENCE_MODEL].set_index("session")["label_hr"]
    out = {}
    for model in MODELS:
        sub = df[df.model == model].copy()
        if sub.empty:
            continue
        sub["ref_hr"] = sub["session"].map(ref)
        if sub["ref_hr"].isna().any():
            sys.exit(f"internal row: {int(sub['ref_hr'].isna().sum())} {model} sessions lack the common reference")
        sub["ae"] = (sub["pred_hr"] - sub["ref_hr"]).abs()
        out[model] = float(sub.groupby("participant_key")["ae"].mean().mean())
    return out


def kr_to_external_row(df: pd.DataFrame) -> Dict[str, float]:
    out = {}
    for model in MODELS:
        sub = df[df.model == model]
        if sub.empty:
            continue
        folds = sub.groupby("fold")["abs_error"].mean()
        if len(folds) != 5:
            print(f"  note: {model} has {len(folds)}/5 folds", file=sys.stderr)
        out[model] = float(folds.mean())
    return out


def session_row(df: pd.DataFrame) -> Dict[str, float]:
    return {model: float(df[df.model == model]["abs_error"].mean()) for model in MODELS if (df.model == model).any()}


def fmt(value: Optional[float]) -> str:
    return f"{value:7.2f}" if value is not None else "      -"


def fmt_expected(value: Optional[float]) -> str:
    return f"({value:5.2f})" if value is not None else "(    -)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=None, help="default: $KR_RPPG_WORK/results")
    args = parser.parse_args(argv)
    results = args.results or work_dir() / "results"
    print(f"{'Table 8 row':34s}" + "".join(f"{DISPLAY[m]:>15s}" for m in MODELS))
    for family, label, expected in ROWS:
        path = results / f"{family}.csv"
        if not path.exists():
            print(f"{label:34s}  (not run: {path.name} missing)")
            continue
        df = pd.read_csv(path, dtype={"participant_key": str, "fold": str, "session": str})
        if family == "kr_kfold_eval":
            values = internal_row(df)
        elif family.startswith("kr_to_"):
            values = kr_to_external_row(df)
        else:
            values = session_row(df)
        reference = expected if expected is not None else (None,) * len(MODELS)
        print(f"{label:34s}" + "".join(f"{fmt(values.get(m))} {fmt_expected(e)}" for m, e in zip(MODELS, reference)))
    print("\nvalues in parentheses: manuscript; see README 'Expected results' for why KR-rPPG test rows differ "
          "when the released (BFDT) videos are used")
    return 0


if __name__ == "__main__":
    sys.exit(main())
