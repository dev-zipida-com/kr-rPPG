#!/usr/bin/env python3
"""Draw Figure 3 (task-stratified MAE with bootstrap 95% CIs) from figure3_values.py output.

    python analysis/figure3/make_figure3.py --values $KR_RPPG_WORK/results/figure3_values.json --out fig3.pdf
    python analysis/figure3/make_figure3.py --values analysis/figure3/expected_values.json --out fig3_ms.png

Bars: mean session-level MAE per model and task (participant mean); error bars: percentile bootstrap 95% CI.
The manuscript figure has a different layout; this matplotlib version shows the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TASKS = {"T00": "Talking", "T01": "Stationary rest", "T02": "Yaw rotation", "T03": "Pitch rotation"}
MODELS = ("PhysNet", "PhysFormer", "DeepPhys", "TS-CAN", "EfficientPhys")
# Colour-blind-safe qualitative palette (Okabe-Ito); spatiotemporal models in blues, frame-difference in warm hues.
COLOURS = {"PhysNet": "#0072B2", "PhysFormer": "#56B4E9", "DeepPhys": "#D55E00", "TS-CAN": "#E69F00",
           "EfficientPhys": "#CC79A7"}


def draw(values: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(TASKS))
    width = 0.8 / len(MODELS)
    for i, model in enumerate(MODELS):
        cells = [values[model][task] for task in TASKS]
        mae = np.array([c["mae"] for c in cells])
        err = np.array([[c["mae"] - c["ci95_lo"] for c in cells], [c["ci95_hi"] - c["mae"] for c in cells]])
        ax.bar(x - 0.4 + width * (i + 0.5), mae, width, yerr=err, capsize=2, label=model,
               color=COLOURS[model], edgecolor="none", error_kw={"elinewidth": 0.8, "ecolor": "#333333"})
    ax.set_xticks(x, list(TASKS.values()))
    ax.set_ylabel("Session-level MAE (bpm)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help=".pdf, .png or .svg")
    args = parser.parse_args()
    draw(json.loads(args.values.read_text())["values"], args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
