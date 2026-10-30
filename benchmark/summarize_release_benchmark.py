#!/usr/bin/env python3
"""Summarize final release-basis prediction CSVs without historical labels/results.

Example (after final prediction collection):
  python benchmark/summarize_release_benchmark.py --results WORK/results \
      --reference WORK/protocol/reference_hr.csv

The default requires the internal family plus all four outgoing and three
incoming families. --families can explicitly select complete families; no
partially collected model/fold data are accepted. This script does not collect
predictions, train models, modify a manuscript, or read legacy OOF label tables.

Internal MAE: mean session AE within participant, then mean over participants.
Outgoing MAE: mean session AE per fold, then mean over all five folds.
Incoming MAE: mean over all released KR sessions for one model per architecture.
Figure 3: participant-mean MAE, with 10,000 participant bootstrap replicates and
a fresh numpy.default_rng(42) per cell; 2.5/97.5 percentile CI. Median AE and
AE >5 bpm percentages use sessions. All scalar references are already fixed by
the release reference tables; no model-specific label HR is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import DISPLAY, MODELS, work_dir

FAMILIES = {
    "kr_kfold": {"aliases": ["kr_kfold", "kr_kfold_eval"], "training": "KR-rPPG", "test": "KR-rPPG", "kind": "internal"},
    "kr_to_pure": {"aliases": ["kr_to_pure"], "training": "KR-rPPG", "test": "PURE", "kind": "outgoing"},
    "kr_to_ubfc_rppg": {"aliases": ["kr_to_ubfc_rppg"], "training": "KR-rPPG", "test": "UBFC-rPPG", "kind": "outgoing"},
    "kr_to_mmpd": {"aliases": ["kr_to_mmpd"], "training": "KR-rPPG", "test": "MMPD", "kind": "outgoing"},
    "kr_to_ubfc_phys": {"aliases": ["kr_to_ubfc_phys", "kr_to_ubfcphys"], "training": "KR-rPPG", "test": "UBFC-Phys", "kind": "outgoing"},
    "pure_to_kr": {"aliases": ["pure_to_kr"], "training": "PURE", "test": "KR-rPPG", "kind": "incoming"},
    "ubfc_rppg_to_kr": {"aliases": ["ubfc_rppg_to_kr"], "training": "UBFC-rPPG", "test": "KR-rPPG", "kind": "incoming"},
    "mmpd_to_kr": {"aliases": ["mmpd_to_kr", "mmpd_to_kr_eval"], "training": "MMPD", "test": "KR-rPPG", "kind": "incoming"},
}
TASK_LABELS = {"T00": "Talking", "T01": "Stationary rest", "T02": "Yaw rotation", "T03": "Pitch rotation"}
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42
AGGREGATION = {
    "internal": "mean session absolute error within each participant, then mean across participants",
    "outgoing": "mean session absolute error within each of five fold models, then arithmetic mean of the five fold means",
    "incoming": "mean absolute error over all released KR-rPPG sessions; one trained checkpoint per architecture",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def metadata(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "sha256": sha256(path)}


def load_reference(path: Path) -> pd.DataFrame:
    frame = read_csv(path)
    required = {"session", "reference_hr", "common_eval_frames"}
    if required - set(frame.columns):
        raise ValueError(f"{path}: missing reference columns {sorted(required - set(frame.columns))}")
    if frame.empty or frame.session.duplicated().any():
        raise ValueError(f"{path}: empty or duplicate reference sessions")
    for col in ("reference_hr", "common_eval_frames"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[["reference_hr", "common_eval_frames"]].to_numpy()).all() or frame.common_eval_frames.le(0).any():
        raise ValueError(f"{path}: invalid reference values or intervals")
    return frame.set_index("session", drop=False)


def resolve_csv(results: Path, family: str) -> Path:
    paths = [results / f"{alias}.csv" for alias in FAMILIES[family]["aliases"]]
    existing = [p for p in paths if p.is_file()]
    if len(existing) != 1:
        raise ValueError(f"{family}: require exactly one prediction CSV, found {[str(p) for p in existing]}")
    return existing[0]


def prepare_family(path: Path, family: str, kr_ref: pd.DataFrame, kr_reference_path: Path, work: Path) -> tuple[pd.DataFrame, dict, dict]:
    definition = FAMILIES[family]
    provenance_path = path.with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("partial_monitoring") is not False:
        raise ValueError(f"{family}: only a final collector run without --allow-partial is accepted")
    if provenance.get("model_specific_label_hr_used") is not False:
        raise ValueError(f"{family}: provenance must explicitly use the independent common reference")
    if bool(provenance.get("oof")) != (definition["kind"] == "internal"):
        raise ValueError(f"{family}: collector OOF scope is inconsistent with the family")
    if definition["kind"] == "outgoing":
        reference_path = Path(provenance["reference_path"]).resolve()
        try:
            reference_path.relative_to(work.resolve())
        except ValueError as error:
            raise ValueError(f"{family}: outgoing reference must be a newly generated file inside WORK: {reference_path}") from error
        reference = load_reference(reference_path)
    else:
        reference_path, reference = kr_reference_path, kr_ref
    if sha256(reference_path) != provenance.get("reference_sha256"):
        raise ValueError(f"{family}: current reference hash differs from the collector's input")

    frame = read_csv(path)
    if "participant_key" not in frame and "pid" in frame:
        frame = frame.rename(columns={"pid": "participant_key"})
    required = {"family", "model", "fold", "session", "participant_key", "task", "pred_hr", "reference_hr", "abs_error", "common_eval_frames", "predicted_frames", "fs"}
    if required - set(frame.columns):
        raise ValueError(f"{family}: missing prediction columns {sorted(required - set(frame.columns))}")
    if frame.empty or set(frame.model) != set(MODELS):
        raise ValueError(f"{family}: final data must contain all five models {MODELS}")
    if not set(frame.family).issubset(set(definition["aliases"])) or frame.family.nunique() != 1:
        raise ValueError(f"{family}: unexpected family labels {sorted(set(frame.family))}")
    if frame.duplicated(["model", "fold", "session"]).any():
        raise ValueError(f"{family}: duplicate model/fold/session predictions")
    numeric = ["pred_hr", "reference_hr", "abs_error", "common_eval_frames", "predicted_frames", "fs"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError(f"{family}: nonfinite numeric predictions")
    if frame.common_eval_frames.le(0).any() or frame.predicted_frames.lt(frame.common_eval_frames).any() or frame.fs.le(0).any():
        raise ValueError(f"{family}: invalid frame coverage or sampling rate")
    for col in ["common_eval_frames", "predicted_frames"]:
        if (frame[col] != np.floor(frame[col])).any():
            raise ValueError(f"{family}: nonintegral frame counts")
    error = (frame.pred_hr - frame.reference_hr).abs()
    max_error_difference = float((error - frame.abs_error).abs().max())
    if max_error_difference > 1e-10:
        raise ValueError(f"{family}: abs_error disagrees with |pred_hr-reference_hr|: {max_error_difference}")
    frame["ae"] = error

    if not set(frame.session).issubset(set(reference.index)):
        raise ValueError(f"{family}: prediction contains an unregistered test session")
    for col in ["reference_hr", "common_eval_frames"]:
        expected = frame.session.map(reference[col]).to_numpy(dtype=float)
        if not np.allclose(frame[col].to_numpy(dtype=float), expected, rtol=0, atol=1e-10):
            raise ValueError(f"{family}: {col} differs from the fixed reference manifest")
    fs_col = "fs" if "fs" in reference else "sampling_rate_hz" if "sampling_rate_hz" in reference else None
    if fs_col:
        expected_fs = frame.session.map(pd.to_numeric(reference[fs_col], errors="raise")).to_numpy(dtype=float)
        if not np.allclose(frame.fs, expected_fs, rtol=0, atol=1e-10):
            raise ValueError(f"{family}: sampling rate differs from the reference")
    for col in ["participant_key", "task"]:
        if col in reference and not frame[col].eq(frame.session.map(reference[col])).all():
            raise ValueError(f"{family}: {col} differs from the fixed reference manifest")
    if definition["kind"] in ("internal", "incoming") and frame.participant_key.eq("").any():
        raise ValueError(f"{family}: released KR participant IDs are required")

    expected_sessions = set(reference.index)
    model_scope = {}
    for model in MODELS:
        subset = frame[frame.model.eq(model)]
        folds = set(subset.fold)
        if definition["kind"] == "internal":
            if folds != set("01234") or len(subset) != len(reference) or set(subset.session) != expected_sessions:
                raise ValueError(f"{family}/{model}: incomplete OOF coverage")
            if not subset.fold.eq(subset.session.map(reference["fold"])).all():
                raise ValueError(f"{family}/{model}: test fold differs from the released fold assignment")
            if subset.groupby("participant_key").fold.nunique().max() != 1:
                raise ValueError(f"{family}/{model}: participant occurs in more than one held-out fold")
            if subset.duplicated(["participant_key", "task"]).any():
                raise ValueError(f"{family}/{model}: duplicate participant/task prediction")
        elif definition["kind"] == "outgoing":
            if folds != set("01234"):
                raise ValueError(f"{family}/{model}: outgoing evaluation requires all five folds")
            for fold, part in subset.groupby("fold", sort=True):
                if len(part) != len(reference) or set(part.session) != expected_sessions:
                    raise ValueError(f"{family}/{model}/fold{fold}: incomplete fixed external test set")
        else:
            if len(folds) != 1 or len(subset) != len(reference) or set(subset.session) != expected_sessions:
                raise ValueError(f"{family}/{model}: incoming evaluation requires one prediction per released KR session")
        model_scope[model] = {"rows": len(subset), "sessions": int(subset.session.nunique()),
                              "participants": int(subset.loc[subset.participant_key.ne(""), "participant_key"].nunique()), "folds": sorted(folds)}
    scope = {"reference_sessions": len(reference), "rows": len(frame), "models": model_scope,
             "abs_error_max_difference": max_error_difference, "fixed_session_set_and_reference_match": True,
             "collector_final_scope": True, "reference": metadata(reference_path)}
    source = {**metadata(path), "collector_provenance": metadata(provenance_path)}
    return frame, scope, source


def aggregate(frame: pd.DataFrame, family: str) -> dict:
    definition = FAMILIES[family]
    models = {}
    for model in MODELS:
        subset = frame[frame.model.eq(model)]
        if definition["kind"] == "internal":
            mae = float(subset.groupby("participant_key", sort=True).ae.mean().mean())
        elif definition["kind"] == "outgoing":
            mae = float(subset.groupby("fold", sort=True).ae.mean().mean())
        else:
            mae = float(subset.ae.mean())
        item = {"display_name": DISPLAY[model], "mae": mae, "n_prediction_rows": len(subset),
                "n_unique_sessions": int(subset.session.nunique()),
                "n_unique_participants": int(subset.loc[subset.participant_key.ne(""), "participant_key"].nunique()),
                "session_pooled_mean": float(subset.ae.mean()), "session_median_ae": float(subset.ae.median()),
                "n_gt5": int(subset.ae.gt(5).sum()), "pct_gt5": float(subset.ae.gt(5).mean() * 100)}
        if definition["kind"] == "outgoing":
            folds = subset.groupby("fold", sort=True).ae.agg(["mean", "count"])
            item["fold_maes"] = {str(k): float(v) for k, v in folds["mean"].items()}
            item["n_sessions_per_fold"] = {str(k): int(v) for k, v in folds["count"].items()}
            item["fold_mae_sample_sd"] = float(folds["mean"].std(ddof=1))
            item["fold_mae_min"] = float(folds["mean"].min())
            item["fold_mae_max"] = float(folds["mean"].max())
            item["tail_denominator_note"] = "n_gt5 and pct_gt5 pool all five fold/session prediction rows, not distinct sessions"
        models[model] = item
    return {"training_set": definition["training"], "test_set": definition["test"],
            "kind": definition["kind"], "aggregation": AGGREGATION[definition["kind"]], "models": models}


def figure_cell(subset: pd.DataFrame) -> dict:
    means = subset.groupby("participant_key", sort=True).ae.mean().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples = rng.integers(0, len(means), size=(BOOTSTRAP_REPLICATES, len(means)))
    bootstrap = means[samples].mean(axis=1)
    low, high = np.percentile(bootstrap, [2.5, 97.5], method="linear")
    return {"n_sessions": len(subset), "n_participants": len(means), "mae": float(means.mean()),
            "ci95_lo": float(low), "ci95_hi": float(high), "mae_session_avg": float(subset.ae.mean()),
            "median_ae": float(subset.ae.median()), "n_gt5": int(subset.ae.gt(5).sum()),
            "pct_gt5": float(subset.ae.gt(5).mean() * 100), "n_ge5": int(subset.ae.ge(5).sum())}


def make_figure(frame: pd.DataFrame, source: dict, kr_ref: pd.DataFrame) -> tuple[dict, list[dict]]:
    values, rows = {}, []
    groups = ["All", *TASK_LABELS]
    for model in MODELS:
        display = DISPLAY[model]
        selected = frame[frame.model.eq(model)]
        values[display] = {}
        for group in groups:
            subset = selected if group == "All" else selected[selected.task.eq(group)]
            if subset.empty:
                raise ValueError(f"Figure3/{display}/{group}: no data")
            cell = figure_cell(subset)
            values[display][group] = cell
            rows.append({"task": group, "task_label": "All tasks" if group == "All" else TASK_LABELS[group],
                         "model": display, "model_id": model,
                         "n_participants": cell["n_participants"], "n_sessions": cell["n_sessions"],
                         "mean_MAE_bpm": cell["mae"], "CI95_lower_bpm": cell["ci95_lo"], "CI95_upper_bpm": cell["ci95_hi"],
                         "session_mean_MAE_bpm": cell["mae_session_avg"], "median_AE_bpm": cell["median_ae"],
                         "n_gt5": cell["n_gt5"], "pct_gt5": cell["pct_gt5"], "n_ge5": cell["n_ge5"]})
    order = [DISPLAY[m] for m in MODELS]
    pct_ranges = {task: [min(values[m][task]["pct_gt5"] for m in order), max(values[m][task]["pct_gt5"] for m in order)] for task in TASK_LABELS}
    rises = {m: values[m]["T03"]["pct_gt5"] - values[m]["T01"]["pct_gt5"] for m in order}
    nonzero_medians = [{"model": m, "task": task, "median_ae": values[m][task]["median_ae"]} for m in order for task in TASK_LABELS if values[m][task]["median_ae"] != 0]
    figure = {
        "schema_version": "release-basis-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__)), "source": source,
        "method": {"reference_hr": "Model-independent released raw BVP over protocol common_eval_frames; same scalar HR for every model; no old label HR read",
                   "session_error": "|pred_hr-reference_hr| in bpm", "cell_mae": AGGREGATION["internal"],
                   "ci": "Percentile bootstrap over participant means sorted by participant_key; B=10000; fresh numpy.random.default_rng(42) per cell; rng.integers(0,n,size=(B,n)); linear percentiles 2.5/97.5",
                   "median_ae": "Median of per-session absolute errors", "large_error": "AE >5 bpm; numerator and percentage over sessions",
                   "numpy": np.__version__, "pandas": pd.__version__, "python": platform.python_version()},
        "structure": {"rows": len(frame), "models": list(MODELS),
                      "rows_per_model": {m: int(frame.model.eq(m).sum()) for m in MODELS},
                      "participants": int(kr_ref.participant_key.nunique()),
                      "sessions_per_task": {task: int(kr_ref.task.eq(task).sum()) for task in TASK_LABELS},
                      "duplicate_model_participant_task_rows": 0, "participants_in_more_than_one_fold": 0,
                      "common_reference_identical_all_models": True},
        "figure_recommendation": {"groups": list(TASK_LABELS), "model_order": order, "model_display_names": order,
                                  "task_labels": TASK_LABELS, "all_group": "Not plotted; retained as the Table8 internal-row anchor"},
        "values": values,
        "results_text_claims": {
            "rotation_tasks_higher_mae_than_T00_T01_all_models": all(min(values[m]["T02"]["mae"], values[m]["T03"]["mae"]) > max(values[m]["T00"]["mae"], values[m]["T01"]["mae"]) for m in order),
            "median_ae_zero_cells": f"{20-len(nonzero_medians)}/20", "median_ae_nonzero_cells": nonzero_medians,
            "pct_gt5_ranges": pct_ranges, "rise_pct_gt5_T01_to_T03_pp": rises,
            "frame_diff_rise_exceeds_spatiotemporal": min(rises[m] for m in ["DeepPhys", "TS-CAN", "EfficientPhys"]) > max(rises[m] for m in ["PhysNet", "PhysFormer"]),
        },
        "plotting_limits": {"largest_task_ci95_hi": max(values[m][task]["ci95_hi"] for m in order for task in TASK_LABELS),
                            "note": "No fixed 0-25 bpm assertion; inspect new CI extent before rendering"},
    }
    return figure, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, help="default: $KR_RPPG_WORK/results")
    parser.add_argument("--reference", type=Path, help="default: WORK/protocol/reference_hr.csv")
    parser.add_argument("--output-dir", type=Path, help="default: the results directory")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES), help="Explicit complete-family subset; default requires all eight families")
    args = parser.parse_args()
    results = (args.results or work_dir() / "results").resolve()
    work = results.parent
    reference_path = (args.reference or work / "protocol/reference_hr.csv").resolve()
    output = (args.output_dir or results).resolve()
    if len(set(args.families)) != len(args.families):
        raise ValueError("Repeated families are not allowed")
    expected_outputs = [output / "benchmark_metrics.json"]
    if "kr_kfold" in args.families:
        expected_outputs += [output / "Figure3_data.csv", output / "figure3_values.json"]
    if any(path.exists() for path in expected_outputs):
        raise FileExistsError("Benchmark summary already exists; choose a new output directory")
    kr_ref = load_reference(reference_path)
    for col in ["participant_key", "task", "fold"]:
        if col not in kr_ref or kr_ref[col].eq("").any():
            raise ValueError(f"KR reference lacks {col}")
    metrics = {"schema_version": "release-basis-v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
               "script": metadata(Path(__file__)), "kr_reference": metadata(reference_path),
               "model_order": list(MODELS), "model_display_order": [DISPLAY[m] for m in MODELS],
               "aggregation_definitions": AGGREGATION,
               "complete_table8_family_set": set(args.families) == set(FAMILIES),
               "missing_families": [f for f in FAMILIES if f not in args.families],
               "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
               "families": {}, "sources": {}, "scope_checks": {}}
    figure, figure_rows = None, None
    for family in args.families:
        path = resolve_csv(results, family)
        frame, scope, source = prepare_family(path, family, kr_ref, reference_path, work)
        metrics["families"][family] = aggregate(frame, family)
        metrics["scope_checks"][family], metrics["sources"][family] = scope, source
        if family == "kr_kfold":
            figure, figure_rows = make_figure(frame, source, kr_ref)
        print(f"{family}: {len(frame)} final prediction rows; {AGGREGATION[FAMILIES[family]['kind']]}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    if figure is not None:
        pd.DataFrame(figure_rows).to_csv(output / "Figure3_data.csv", index=False)
        (output / "figure3_values.json").write_text(json.dumps(figure, indent=2, allow_nan=False) + "\n")
        metrics["figure3_outputs"] = {p.name: metadata(p) for p in [output / "Figure3_data.csv", output / "figure3_values.json"]}
    (output / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(f"Saved {output / 'benchmark_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
