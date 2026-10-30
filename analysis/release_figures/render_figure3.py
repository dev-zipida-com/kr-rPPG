#!/usr/bin/env python3
"""Render release-based participant MAEs using the established Figure 3 design.

Statistics must be supplied by summarize_release_benchmark.py. No historical
model result is read or used as a fallback.
"""
from __future__ import annotations

import argparse
import csv
import math
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = None
STEM = "Figure_3_task_MAE"
WIDTH_MM, HEIGHT_MM = 180.0, 114.0
TEXT = "#222222"
ACCENT = "#000000"
MUTED = "#7c8790"
GRID = "#e2e6e8"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    global HERE, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    SOURCE = args.values.resolve()
    HERE = args.output.resolve()
    HERE.mkdir(parents=True, exist_ok=True)
    source = json.loads(SOURCE.read_text())
    if (source.get('schema_version') != 'release-basis-v1' or
            source.get('structure', {}).get('common_reference_identical_all_models') is not True):
        raise ValueError('Release-based results with a common raw-BVP reference are required')
    tasks = ["T00", "T01", "T02", "T03"]
    models = ["PhysNet", "PhysFormer", "DeepPhys", "TS-CAN", "EfficientPhys"]
    task_labels = source["figure_recommendation"]["task_labels"]
    assert models == source["figure_recommendation"]["model_order"]
    maximum_ci = max(source['values'][model][task]['ci95_hi'] for model in models for task in tasks)
    axis_upper = max(25, 5 * math.ceil(maximum_ci / 5))
    rows = []
    for task in tasks:
        for model in models:
            value = source["values"][model][task]
            assert 0 <= value["ci95_lo"] <= value["mae"] <= value["ci95_hi"] <= axis_upper
            assert value["n_participants"] == source["structure"]["sessions_per_task"][task]
            rows.append({
                "task": task, "task_label": task_labels[task], "model": model,
                "n_participants": value["n_participants"],
                "n_sessions": value["n_sessions"],
                "mean_MAE_bpm": value["mae"],
                "CI95_lower_bpm": value["ci95_lo"],
                "CI95_upper_bpm": value["ci95_hi"],
            })
    with (HERE / "Figure_3_plotted_data.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.3,
        "text.color": TEXT, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "svg.hashsalt": "KR-rPPG-Figure3-20261023",
    })
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    axes = []
    for index, task in enumerate(tasks):
        column = index % 2
        row = index // 2
        panel_left = 5 + 90 * column
        plot_left = 31 + 90 * column
        plot_bottom = 66 - 56 * row
        ax = fig.add_axes([plot_left / WIDTH_MM, plot_bottom / HEIGHT_MM,
                           54 / WIDTH_MM, 30 / HEIGHT_MM])
        axes.append(ax)
        y = np.arange(len(models))[::-1]
        means = np.array([source["values"][m][task]["mae"] for m in models])
        lower = np.array([source["values"][m][task]["ci95_lo"] for m in models])
        upper = np.array([source["values"][m][task]["ci95_hi"] for m in models])
        ax.errorbar(means, y, xerr=np.stack([means - lower, upper - means]),
                    fmt="o", markersize=3.9, color=ACCENT, ecolor=ACCENT,
                    elinewidth=0.95, capsize=2.1, capthick=0.8, zorder=3)
        ax.set_xlim(0, axis_upper)
        ax.set_ylim(-0.6, 4.6)
        ax.set_yticks(y, models)
        ax.set_xticks(np.arange(0, axis_upper + 1, 5 if axis_upper <= 40 else 10))
        ax.tick_params(axis="x", length=2.5, width=0.6, pad=3.5, labelsize=8)
        ax.tick_params(axis="y", length=0, pad=6, labelsize=8.3)
        ax.set_xlabel("MAE (bpm)", fontsize=8.3, labelpad=4.5)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color=GRID, linewidth=0.45)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        fig.text(panel_left / WIDTH_MM, (107 - 56 * row) / HEIGHT_MM,
                 "abcd"[index], fontsize=10, fontweight="bold", ha="left", va="baseline")
        fig.text(plot_left / WIDTH_MM, (107 - 56 * row) / HEIGHT_MM,
                 f"{task}  {task_labels[task]}", fontsize=9, fontweight="bold",
                 ha="left", va="baseline")
        n = source["structure"]["sessions_per_task"][task]
        fig.text(plot_left / WIDTH_MM, (101 - 56 * row) / HEIGHT_MM,
                 f"n = {n} participants", fontsize=8, color="#525c63",
                 ha="left", va="baseline")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    outside = []
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            bbox = artist.get_window_extent(renderer)
            if (bbox.x0 < canvas.x0 or bbox.x1 > canvas.x1 or
                    bbox.y0 < canvas.y0 or bbox.y1 > canvas.y1):
                outside.append(artist.get_text())
    assert not outside, f"Text outside the figure: {outside}"
    metadata = {"Title": "KR-rPPG task-stratified rPPG estimation error",
                "Author": "KR-rPPG", "Subject": "Verified mean MAE and participant-bootstrap 95% confidence intervals"}
    fig.savefig(HERE / f"{STEM}.pdf", metadata=metadata)
    fig.savefig(HERE / f"{STEM}.svg", metadata={
        "Title": metadata["Title"],
        "Description": f"Points show mean MAE; horizontal intervals show participant-bootstrap 95% confidence intervals. Same 0–{axis_upper} bpm scale in all panels. Actual released-video results."
    })
    fig.savefig(HERE / f"{STEM}.png", dpi=600)
    fig.savefig(HERE / f"{STEM}_preview.png", dpi=180)
    plt.close(fig)

    raw_path = Path(source["source"]["path"])
    actual_raw_sha256 = sha256(raw_path) if raw_path.exists() else None
    if actual_raw_sha256 is not None:
        assert actual_raw_sha256 == source["source"]["sha256"]
    outputs = [f"{STEM}.{extension}" for extension in ["pdf", "svg", "png"]]
    outputs += [f"{STEM}_preview.png", "Figure_3_plotted_data.csv"]
    manifest = {
        "figure": "Figure 3", "design": "Four task panels; identical model order; mean points and horizontal 95% confidence intervals",
        "size_mm": [WIDTH_MM, HEIGHT_MM], "shared_MAE_axis_bpm": [0, axis_upper],
        "font": "DejaVu Sans", "model_order": models, "task_order": tasks,
        "renderer_path": str(Path(__file__).resolve()), "renderer_sha256": sha256(Path(__file__)),
        "aggregate_snapshot": str(SOURCE), "aggregate_sha256": sha256(SOURCE),
        "raw_source": source["source"], "raw_source_sha256_verified": actual_raw_sha256,
        "statistical_method": source["method"],
        "statistics_recomputed": False, "plotted_rows": len(rows),
        "statistics_source": "Completed release-benchmark aggregate; renderer copies its values and intervals",
        "original_video_results": False, "release_format_results": True,
        "validation": {"all_intervals_on_common_axis": True, "no_text_outside_canvas": True,
                       "plotted_values_copied_without_rounding": True},
        "software": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "files": {name: {"sha256": sha256(HERE / name), "bytes": (HERE / name).stat().st_size} for name in outputs},
    }
    (HERE / "Figure_3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_directory": str(HERE), "files": outputs,
                      "plotted_rows": len(rows), "raw_sha256_verified": actual_raw_sha256}, indent=2))


if __name__ == "__main__":
    main()
