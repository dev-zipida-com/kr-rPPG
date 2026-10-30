#!/usr/bin/env python3
"""Regenerate metadata, synchronized-signal and Table 7 summaries from release inputs.

All KR inputs are released files or build_reference_hr.py outputs. Original
study participant mappings, OOF labels, caches and excluded sessions are not read.
External Table 7 references retain their original public-dataset basis.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import platform
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.io import loadmat
from scipy.signal import find_peaks


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path, manifest: list, scope: str = "release") -> pd.DataFrame:
    payload = path.read_bytes()
    manifest.append({"scope": scope, "path": str(path), "sha256": digest(payload), "bytes": len(payload)})
    frame = pd.read_csv(io.BytesIO(payload), float_precision="round_trip")
    frame.attrs["input_sha256"] = digest(payload)
    return frame


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def describe(values) -> dict:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    q1, median, q3 = np.percentile(x, [25, 50, 75])
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
            "median": float(median), "q1": float(q1), "q3": float(q3), "min": float(x.min()), "max": float(x.max())}


def count_pct(n: int, total: int) -> dict:
    return {"n": int(n), "denominator": int(total), "percent": float(100 * n / total)}


def make_metadata(p: pd.DataFrame, availability: pd.DataFrame, dictionary: pd.DataFrame) -> dict:
    n = len(p)
    missing = {c: int(p[c].notna().sum()) for c in p.columns}
    dictionary_availability = dict(zip(dictionary.variable, dictionary.availability_n_of_296.astype(int)))
    if missing != dictionary_availability:
        raise ValueError("Released dictionary availability differs from observed metadata")
    any_comorbidity = (p.dm.eq(1) | p.htn.eq(1) | p.dyslipidemia.eq(1) | p.other_comorbidity_count.gt(0))
    age_groups = pd.cut(p.age, [-np.inf, 49, 59, 69, 79, np.inf], labels=["<=49", "50-59", "60-69", "70-79", ">=80"])
    clinical = {c: count_pct(int(p[c].eq(1).sum()), int(p[c].notna().sum())) for c in ["dm", "htn", "dyslipidemia", "ild"]}
    clinical["any_comorbidity"] = count_pct(int(any_comorbidity.sum()), n)
    bmi = p.bmi.dropna()
    sensitivity = bmi[bmi < 100]
    extreme = p.loc[p.bmi.ge(100), ["participant_key", "height", "weight", "bmi"]]
    return {
        "participants": n, "columns": len(p.columns), "column_order": list(p.columns),
        "methods": {"sd": "sample SD, ddof=1", "missingness": "nonmissing denominator for each variable; no imputation",
                    "outliers": "All stored values are included in the primary summaries; no correction or exclusion",
                    "any_comorbidity": "dm==1 OR htn==1 OR dyslipidemia==1 OR other_comorbidity_count>0; no separate ILD term",
                    "other_comorbidity_count": "Legacy column name; binary presence indicator: 0=none, 1=at least one of the five specified conditions; missing values are not imputed",
                    "bmi_sensitivity": "Descriptive sensitivity omitting stored BMI >=100 kg/m²; not the primary analysis and not a data correction",
                    "quantiles": "numpy percentile, linear interpolation"},
        "age": describe(p.age), "age_ge60": count_pct(int(p.age.ge(60).sum()), n),
        "age_groups": {str(k): count_pct(int(v), n) for k, v in age_groups.value_counts(sort=False).items()},
        "sex": {k: count_pct(v, n) for k, v in Counter(p.sex).items()},
        "site": {k: count_pct(v, n) for k, v in Counter(p.site).items()},
        "clinical_prevalence": clinical, "bmi_primary": describe(bmi),
        "bmi_sensitivity_lt100": describe(sensitivity), "bmi_extreme_stored_records": extreme.to_dict(orient="records"),
        "continuous_variables": {c: describe(p[c]) for c in ["age", "height", "weight", "bmi", "sbp", "dbp", "hr_rest", "hba1c", "fpg", "creatinine", "gfr", "ast", "alt"]},
        "availability": missing, "dictionary_availability_matches": True,
        "other_comorbidity_count_observed": sorted(int(x) for x in p.other_comorbidity_count.dropna().unique()),
        "other_comorbidity_indicator_distribution": {str(int(k)): int(v) for k, v in p.other_comorbidity_count.dropna().value_counts().sort_index().items()},
        "sessions_by_task": {f"T0{i}": int(availability[f"t0{i}_released"].sum()) for i in range(4)},
        "complete_four_task_participants": int(availability[[f"t0{i}_released" for i in range(4)]].eq(1).all(axis=1).sum()),
    }


def one_session(root: Path, ref: pd.Series, person: pd.Series, eligibility: pd.Series, inputs: list) -> dict:
    directory = root / ref.participant_key / ref.task
    frame = read_csv(directory / "frames_timestamp.csv", inputs)
    arrays, bvp_hash = {}, None
    for filename, col in [("BVP_synchronized.csv", "bvp"), ("HR_synchronized.csv", "hr"), ("SpO2_synchronized.csv", "spo2")]:
        df = read_csv(directory / filename, inputs)
        if len(df) != len(frame) or not np.array_equal(df.timestamp.to_numpy(), frame.timestamp.to_numpy()):
            raise ValueError(f"Signal/frame alignment differs: {ref.session}/{filename}")
        arrays[col] = df[col].to_numpy(dtype=float)
        if col == "bvp":
            bvp_hash = df.attrs["input_sha256"]
    bvp, hr, spo2 = arrays["bvp"], arrays["hr"], arrays["spo2"]
    if bvp_hash != ref.bvp_file_sha256:
        raise ValueError(f"BVP differs from P1 reference input: {ref.session}")
    if not np.isfinite(hr).all() or not np.isfinite(spo2).all():
        raise ValueError(f"Nonfinite device signal: {ref.session}")
    selected = bvp[int(ref.eval_start_frame):int(ref.eval_end_frame_exclusive)]
    if digest(selected.astype("<f8").tobytes()) != ref.benchmark_bvp_float64_le_sha256:
        raise ValueError(f"BVP benchmark interval differs from P1: {ref.session}")
    nwin, valid = len(bvp) // 240, 0
    for start in range(0, nwin * 240, 240):
        window = bvp[start:start + 240]
        sd = np.std(window)
        peaks, _ = find_peaks(window, distance=12, prominence=0.3 * sd if sd > 0 else None)
        valid += 4 <= len(peaks) <= 24
    dt = np.diff(frame.timestamp.to_numpy(dtype=float))
    return {
        "session": ref.session, "session_id": ref.session_id, "participant_key": ref.participant_key,
        "task": ref.task, "fold": int(ref.fold), "site": person.site, "age": float(person.age),
        "n_frames": len(frame), "video_frames_header": int(eligibility.video_frames_header),
        "bbox_rows": int(eligibility.bbox_rows), "common_eval_frames": int(ref.common_eval_frames),
        "benchmark_reference_hr": float(ref.reference_hr),
        "table7_reference_hr": float(ref.table7_reference_hr) if ref.task == "T01" else None,
        "bvp_nonfinite": int((~np.isfinite(bvp)).sum()), "bvp_zero": int((bvp == 0).sum()), "bvp_127": int((bvp == 127).sum()),
        "bvp_windows": nwin, "bvp_valid_windows": valid,
        "bvp_valid_window_fraction": valid / nwin if nwin else None,
        "bvp_discarded_remainder_samples": len(bvp) % 240,
        "device_hr_mean": float(hr.mean()), "device_hr_sd": float(hr.std(ddof=1)),
        "device_hr_plausible_samples_40_200": int(((hr >= 40) & (hr <= 200)).sum()),
        "spo2_mean": float(spo2.mean()), "spo2_sd": float(spo2.std(ddof=1)), "spo2_min": float(spo2.min()),
        "spo2_plausible_samples_70_100": int(((spo2 >= 70) & (spo2 <= 100)).sum()),
        "spo2_any_below95": int((spo2 < 95).any()),
        "interval_n": len(dt), "interval_sum_s": float(dt.sum()),
        "mean_interval_s": float(dt.mean()), "max_interval_s": float(dt.max()),
        "intervals_gt1s": int((dt > 1).sum()), "nonpositive_timestamp_intervals": int((dt <= 0).sum()),
        "nonincreasing_frame_ids": int((np.diff(frame.frame.to_numpy()) <= 0).sum()),
        "mean_interval_outside5pct": int(abs(dt.mean() - 1 / 30) / (1 / 30) > 0.05),
    }


def summarize_signals(m: pd.DataFrame) -> dict:
    t01 = m[m.task.eq("T01")]
    ages = {}
    for name in ["device_hr_mean", "spo2_mean"]:
        valid = t01[["age", name]].dropna()
        young, older = valid.loc[valid.age < 60, name], valid.loc[valid.age >= 60, name]
        spearman = stats.spearmanr(valid.age, valid[name])
        mwu = stats.mannwhitneyu(young, older, alternative="two-sided", method="auto")
        ages[name] = {"n": len(valid), "spearman_r": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                      "age_lt60": describe(young), "age_ge60": describe(older), "mann_whitney_u": float(mwu.statistic), "mann_whitney_p": float(mwu.pvalue)}
    pivot = m.pivot(index="participant_key", columns="task", values="benchmark_reference_hr")
    complete = pivot.dropna(subset=["T00", "T01", "T02", "T03"])
    difference = complete.T00 - complete.T01
    paired = stats.wilcoxon(complete.T00, complete.T01, alternative="two-sided", zero_method="wilcox", correction=False, method="auto")
    by_site = {site: {"n_sessions": len(s), "mean_of_session_mean_spo2": float(s.spo2_mean.mean()),
                      "sessions_any_spo2_below95": count_pct(int(s.spo2_any_below95.sum()), len(s))}
               for site, s in m.groupby("site", sort=True)}
    anomalous = m.loc[m.mean_interval_outside5pct.eq(1) | m.intervals_gt1s.gt(0) | m.nonpositive_timestamp_intervals.gt(0) | m.nonincreasing_frame_ids.gt(0),
                       ["session", "mean_interval_s", "max_interval_s", "intervals_gt1s", "nonpositive_timestamp_intervals", "nonincreasing_frame_ids", "mean_interval_outside5pct"]]
    return {
        "methods": {
            "signal_unit": "Frame-aligned released values; these are not independent raw device samples",
            "descriptive_sd": "ddof=1, including within-session device HR SD",
            "session_weighting": "Equal weight per session for mean SpO2 and mean inter-frame interval; also report pooled sample interval separately",
            "bvp_peaks": "240-sample nonoverlapping windows at nominal 30 Hz; discard remainder; scipy.find_peaks(distance=12,prominence=0.3*population_SD or None if zero); valid 4<=peak count<=24",
            "bvp_nonfinite": "Only nonfinite is classified as missing/nonfinite. Values equal to zero/127 are descriptive checks, not a general validity test after filtering",
            "plausibility_bounds": "Inclusive device HR 40<=HR<=200 for T01; inclusive SpO2 70<=SpO2<=100 across all sessions",
            "physiology_reference": "P1 common-interval benchmark reference_hr, not Table7 reference and not device HR",
            "complete_tasks": "Participants with one released eligible session for all T00,T01,T02,T03",
            "paired_test": "two-sided scipy.stats.wilcoxon on complete participants; zero_method=wilcox, correction=False, method=auto",
            "age_tests": "T01 per-session device HR and SpO2 means joined to released age; Spearman and two-sided Mann-Whitney U, method=auto; no multiplicity adjustment for the descriptive age checks",
            "camera_frame_drop": "Not estimated. Historical synchronization NaN rows do not establish camera-drop rates. Timestamp intervals and frame ID resets are reported directly",
        },
        "n_sessions": len(m), "n_frame_aligned_samples": int(m.n_frames.sum()),
        "bvp": {"n_nonfinite": int(m.bvp_nonfinite.sum()), "n_zero": int(m.bvp_zero.sum()), "n_127": int(m.bvp_127.sum()),
                "valid_windows": count_pct(int(m.bvp_valid_windows.sum()), int(m.bvp_windows.sum())),
                "sessions_ge95pct_valid": count_pct(int(m.bvp_valid_window_fraction.ge(0.95).sum()), len(m)),
                "discarded_remainder_samples": int(m.bvp_discarded_remainder_samples.sum())},
        "t01_device_hr": {"n_sessions": len(t01), "sample_plausibility_40_200": count_pct(int(t01.device_hr_plausible_samples_40_200.sum()), int(t01.n_frames.sum())),
                          "within_session_sd_median": float(t01.device_hr_sd.median()), "within_session_sd_mean": float(t01.device_hr_sd.mean())},
        "spo2": {"sample_plausibility_70_100": count_pct(int(m.spo2_plausible_samples_70_100.sum()), int(m.n_frames.sum())),
                 "mean_of_session_means": float(m.spo2_mean.mean()), "by_site": by_site,
                 "sites_A_B_pooled_session_mean": float(m.loc[m.site.isin(["A", "B"]), "spo2_mean"].mean())},
        "timestamps": {"mean_of_session_mean_intervals_ms": float(m.mean_interval_s.mean() * 1000),
                       "sd_of_session_mean_intervals_ms": float(m.mean_interval_s.std(ddof=1) * 1000),
                       "sample_weighted_mean_interval_ms": float(m.interval_sum_s.sum() / m.interval_n.sum() * 1000),
                       "n_sessions_mean_interval_outside5pct": int(m.mean_interval_outside5pct.sum()),
                       "n_sessions_with_interval_gt1s": int(m.intervals_gt1s.gt(0).sum()), "max_interval_s": float(m.max_interval_s.max()),
                       "n_sessions_with_nonpositive_timestamp_intervals": int(m.nonpositive_timestamp_intervals.gt(0).sum()),
                       "n_sessions_with_nonincreasing_frame_ids": int(m.nonincreasing_frame_ids.gt(0).sum()),
                       "anomalous_sessions": anomalous.to_dict(orient="records")},
        "bbox": {"short_sessions": int(m.bbox_rows.lt(m.n_frames).sum()), "missing_rows_relative_to_frames": int((m.n_frames - m.bbox_rows).clip(lower=0).sum()),
                 "note": "Counts do not establish positions of absent box rows; annotations do not limit benchmark frames"},
        "benchmark_reference_hr": {"all_sessions": describe(m.benchmark_reference_hr),
                                   "sessions_ge90": count_pct(int(m.benchmark_reference_hr.ge(90).sum()), len(m)),
                                   "by_task_all_available": {task: describe(s.benchmark_reference_hr) for task, s in m.groupby("task", sort=True)},
                                   "complete_participants": len(complete),
                                   "by_task_complete_participants": {task: describe(complete[task]) for task in ["T00", "T01", "T02", "T03"]},
                                   "paired_t00_minus_t01": {**describe(difference), "wilcoxon_statistic": float(paired.statistic), "wilcoxon_p": float(paired.pvalue)}},
        "age_associations_t01": ages,
    }


def table7(args, reference: pd.DataFrame, out: Path) -> dict:
    recipe_path = Path(__file__).resolve().parents[1] / "table7" / "table7_hr_distributions.py"
    spec = importlib.util.spec_from_file_location("table7_recipe", recipe_path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    manifest, recordings, participant_values = [], [], []
    kr = reference.loc[reference.task.eq("T01")]
    for row in kr.itertuples():
        participant_values.append({"dataset": "KR-rPPG", "participant": row.participant_key, "hr": float(row.table7_reference_hr), "n_recordings": 1})
    if args.kr_only:
        summary = describe(kr.table7_reference_hr)
        if summary["n"] != 293:
            raise ValueError(f"Unexpected KR Table 7 population: {summary['n']}")
        pd.DataFrame(participant_values).to_csv(out / "table7_participant_values.csv", index=False)
        save_json(out / "table7_input_manifest.json", {
            "scope": "kr_only", "reference": "reference_hr.csv:table7_reference_hr", "external_inputs": []})
        return {"scope": "kr_only", "methods": {
                    "kr": "Full released T01 BVP via protocol/reference_hr.csv:table7_reference_hr; no skip or common benchmark trim",
                    "estimator": "Order-1 Butterworth 0.75-2.5 Hz, filtfilt, no detrending; whole waveform periodogram nfft next power of two, detrend=False; maximum in inclusive band times 60",
                    "description": "Sample SD ddof=1; numpy linear percentile interpolation"},
                "rows": {"KR-rPPG": summary}, "comparisons": {},
                "selection_counts": {"KR-rPPG_T01": len(kr)}}
    pure_files = sorted(p for p in args.pure_root.glob("*-01/*-01.json") if re.fullmatch(r"\d{2}-01", p.stem) and p.parent.name == p.stem)
    for path in pure_files:
        payload = path.read_bytes()
        packages = json.loads(payload)["/FullPackage"]
        seconds = np.asarray([p["Timestamp"] for p in packages], dtype=float) / 1e9
        fs = (len(seconds) - 1) / (seconds[-1] - seconds[0])
        x = np.asarray([p["Value"]["waveform"] for p in packages], dtype=float)
        hr = recipe.reference_hr(x, fs)
        manifest.append({"dataset": "PURE", "path": str(path), "sha256": digest(payload), "selection": "steady session 01", "fs": float(fs), "n_samples": len(x)})
        recordings.append({"dataset": "PURE", "recording": path.stem, "participant": path.stem.split("-")[0], "hr": hr})
        participant_values.append({"dataset": "PURE", "participant": path.stem.split("-")[0], "hr": hr, "n_recordings": 1})
    mmpd_by_subject = defaultdict(list)
    for path in sorted(args.mmpd_root.glob("subject*/*.mat")):
        subject = re.fullmatch(r"subject(\d+)", path.parent.name)
        if not subject:
            continue
        mat = loadmat(str(path), variable_names=["GT_ppg", "motion"])
        motion = str(np.asarray(mat["motion"]).ravel()[0]).strip()
        entry = {"dataset": "MMPD", "path": str(path), "motion": motion, "selected": motion == "Stationary", "file_size_bytes": path.stat().st_size}
        if motion == "Stationary":
            x = np.asarray(mat["GT_ppg"], dtype="<f8").ravel()
            entry.update({"effective_signal_sha256": digest(x.tobytes()), "signal_hash_encoding": "flattened little-endian float64 GT_ppg", "n_samples": len(x), "fs": 30.0})
            hr = recipe.reference_hr(x, 30.0)
            mmpd_by_subject[subject.group(1)].append(hr)
            recordings.append({"dataset": "MMPD", "recording": f"{path.parent.name}/{path.name}", "participant": subject.group(1), "hr": hr})
        manifest.append(entry)
    for subject, values in sorted(mmpd_by_subject.items(), key=lambda z: int(z[0])):
        participant_values.append({"dataset": "MMPD", "participant": subject, "hr": float(np.mean(values)), "n_recordings": len(values)})
    ubfc_files = sorted(args.ubfc_phys_root.glob("s*/bvp_s*_T1.csv"))
    for path in ubfc_files:
        if not re.fullmatch(r"s\d+", path.parent.name) or path.name != f"bvp_{path.parent.name}_T1.csv":
            continue
        payload = path.read_bytes()
        x = pd.read_csv(io.BytesIO(payload), header=None).iloc[:, 0].to_numpy(dtype=float)
        hr = recipe.reference_hr(x, 64.0)
        manifest.append({"dataset": "UBFC-Phys", "path": str(path), "sha256": digest(payload), "selection": "resting T1", "fs": 64.0, "n_samples": len(x)})
        recordings.append({"dataset": "UBFC-Phys", "recording": path.name, "participant": path.parent.name, "hr": hr})
        participant_values.append({"dataset": "UBFC-Phys", "participant": path.parent.name, "hr": hr, "n_recordings": 1})
    v = pd.DataFrame(participant_values)
    rows = {name: describe(s.hr) for name, s in v.groupby("dataset", sort=False)}
    if {k: x["n"] for k, x in rows.items()} != {"KR-rPPG": 293, "PURE": 10, "MMPD": 33, "UBFC-Phys": 56}:
        raise ValueError(f"Unexpected Table 7 populations: {rows}")
    external = {name: v.loc[v.dataset.eq(name), "hr"].to_numpy() for name in ["MMPD", "PURE", "UBFC-Phys"]}
    comparisons = recipe.compare_with_kr(kr.table7_reference_hr.to_numpy(), external)
    v.to_csv(out / "table7_participant_values.csv", index=False)
    pd.DataFrame(recordings).to_csv(out / "table7_external_recordings.csv", index=False)
    save_json(out / "table7_input_manifest.json", {"recipe_path": str(recipe_path), "recipe_sha256": digest(recipe_path.read_bytes()),
              "effective_input_hash_note": "MMPD whole MAT video payloads are not rehashed; selection metadata for every file and float64 reference-waveform hashes for selected recordings identify the consumed statistical inputs", "inputs": manifest})
    return {"methods": {"kr": "Full released T01 BVP via protocol/reference_hr.csv:table7_reference_hr; no skip or common benchmark trim",
                        "external_estimator": "Order-1 Butterworth 0.75-2.5 Hz, filtfilt, no detrending; whole waveform periodogram nfft next power of two, detrend=False; maximum in inclusive band times 60",
                        "selection": "PURE steady session 01; MMPD motion exactly Stationary, recording HR averaged per subject; UBFC-Phys all resting T1",
                        "tests": "Two-sided Welch t-test and Mann-Whitney U; Holm across three KR-versus-external comparisons separately within each test family",
                        "description": "Sample SD ddof=1; numpy linear percentile interpolation"},
            "rows": rows, "comparisons": comparisons,
            "selection_counts": {"KR-rPPG_T01": len(kr), "PURE_steady": len(pure_files), "MMPD_all_files": sum(r["dataset"] == "MMPD" for r in manifest),
                                 "MMPD_stationary_recordings": sum(len(x) for x in mmpd_by_subject.values()), "MMPD_subjects": len(mmpd_by_subject), "UBFC_Phys_T1": len(ubfc_files)},
            "interpretation_flags": {"all_welch_p_ge_0_05": all(r["welch_p"] >= 0.05 for r in comparisons.values()),
                                     "pure_mwu_holm_below_0_05": comparisons["PURE"]["mw_holm"] < 0.05}}


def before_after(metadata: dict, signal: dict, table: dict) -> pd.DataFrame:
    rows = []
    def add(location, metric, old, new, n, why, display=None):
        rows.append({"location": location, "metric": metric, "before_manuscript": old,
                     "after_unrounded": new, "after_display": display if display is not None else new,
                     "denominator_or_n": n, "basis_or_change_reason": why})
    add("Abstract/Table1/Background", "age_mean_sd", "63.8 +/- 12.0", f"{metadata['age']['mean']} +/- {metadata['age']['sd']}", 296, "released age", f"{metadata['age']['mean']:.1f} +/- {metadata['age']['sd']:.1f}")
    for label, old in [("dm", 75.0), ("htn", 44.6), ("dyslipidemia", 37.2), ("any_comorbidity", 87.8)]:
        val = metadata["clinical_prevalence"][label]
        add("Table2", label + "_percent", old, val["percent"], val["denominator"], "released clinical snapshot; any_comorbidity definition unchanged", f"{val['n']}/{val['denominator']} ({val['percent']:.1f}%)")
    for field, old in [("mean", 24.8), ("sd", 4.0)]:
        add("Table2", "bmi_" + field, old, metadata["bmi_primary"][field], metadata["bmi_primary"]["n"], "all released BMI values included, including stored BMI155", f"{metadata['bmi_primary'][field]:.1f}")
    for c, n in metadata["availability"].items():
        add("Table6", c + "_available_n", n, n, 296, "matches released dictionary and observed nonmissing count")
    add("Data integrity", "camera_frame_drop_rate_percent", 1.81, "not estimated", 1166, "historical n_nan_frames is not an established camera-drop definition")
    add("Data integrity", "sessions_without_camera_drops_percent", 57.8, "not estimated", 1166, "report observed timestamp integrity instead")
    ts = signal["timestamps"]
    for field, old in [("mean_of_session_mean_intervals_ms", 33.41), ("sd_of_session_mean_intervals_ms", 1.32), ("n_sessions_mean_interval_outside5pct", 0)]:
        add("Data integrity", field, old, ts[field], 1166, "released timestamp differences")
    add("Data integrity", "bvp_invalid_missing_percent", "<0.70", 0, signal["n_frame_aligned_samples"], "endpoint now explicitly nonfinite released BVP values, not historical raw ADC saturation")
    h = signal["t01_device_hr"]
    add("Data integrity", "t01_device_hr_plausible_percent", 99.51, h["sample_plausibility_40_200"]["percent"], h["sample_plausibility_40_200"]["denominator"], "released frame-aligned device HR; inclusive40-200", f"{h['sample_plausibility_40_200']['percent']:.2f}")
    for field, old in [("within_session_sd_median", 1.46), ("within_session_sd_mean", 1.72)]:
        add("Data integrity", field, old, h[field], 293, "released device HR sample SD within session", f"{h[field]:.2f}")
    for field, old in [("valid_windows", 99.2), ("sessions_ge95pct_valid", 96.1)]:
        val = signal["bvp"][field]
        add("Data integrity", "bvp_" + field + "_percent", old, val["percent"], val["denominator"], "unchanged8s peak detector", f"{val['percent']:.1f}")
    val = signal["spo2"]["sample_plausibility_70_100"]
    add("Data integrity", "spo2_plausible_percent", 99.9, val["percent"], val["denominator"], "released frame-aligned SpO2; inclusive70-100", f"{val['percent']:.1f}")
    hr = signal["benchmark_reference_hr"]
    add("Background/Usage", "sessions_reference_hr_ge90_n", 321, hr["sessions_ge90"]["n"], 1166, "new model-independent common-interval release BVP reference")
    add("Background/Usage", "sessions_reference_hr_ge90_percent", 27.5, hr["sessions_ge90"]["percent"], 1166, "new model-independent common-interval release BVP reference", f"{hr['sessions_ge90']['percent']:.1f}")
    old_tasks = {"T00": (84.9, 14.2), "T01": (80.9, 13.6), "T02": (80.5, 13.6), "T03": (81.2, 14.0)}
    for task, (oldmean, oldsd) in old_tasks.items():
        for field, old in [("mean", oldmean), ("sd", oldsd)]:
            value = hr["by_task_complete_participants"][task][field]
            add("Physiological plausibility", task + "_reference_hr_" + field, old, value, hr["complete_participants"], "new common-interval reference in complete four-task participants", f"{value:.1f}")
    for field, old in [("mean", 4.1), ("sd", 4.3), ("wilcoxon_p", "<1e-39")]:
        add("Physiological plausibility", "paired_t00_minus_t01_" + field, old, hr["paired_t00_minus_t01"][field], hr["complete_participants"], "same complete cohort; new reference")
    spo = signal["spo2"]
    for name, old, value, n in [("mean_spo2", 96.3, spo["mean_of_session_means"], 1166), ("site_C_mean_spo2", 95.9, spo["by_site"]["C"]["mean_of_session_mean_spo2"], 430), ("sites_AB_mean_spo2", 96.6, spo["sites_A_B_pooled_session_mean"], 736), ("site_C_sessions_any_below95_percent", 31.6, spo["by_site"]["C"]["sessions_any_spo2_below95"]["percent"], 430)]:
        add("Physiological plausibility", name, old, value, n, "released frame-aligned SpO2; session weighting", f"{value:.1f}")
    age_old = {"device_hr_mean": (-0.225, 0.0001, 84.1, 11.4, 78.8, 15.0, 0.002), "spo2_mean": (-0.211, 0.0003, 96.6, 1.4, 95.9, 1.7, "<0.0001")}
    for endpoint, olds in age_old.items():
        a = signal["age_associations_t01"][endpoint]
        vals = [a["spearman_r"], a["spearman_p"], a["age_lt60"]["mean"], a["age_lt60"]["sd"], a["age_ge60"]["mean"], a["age_ge60"]["sd"], a["mann_whitney_p"]]
        for field, old, value in zip(["spearman_r", "spearman_p", "lt60_mean", "lt60_sd", "ge60_mean", "ge60_sd", "mwu_p"], olds, vals):
            add("Physiological plausibility", endpoint + "_age_" + field, old, value, 293, "released age and T01 device mean; preserved endpoint")
    old_table = {"KR-rPPG": [293, 81.0, 13.8, 80.4, 70.8, 90.1, 52.7, 130.1], "MMPD": [33, 76.8, 11.7, 75.6, 70.5, 82.8, 51.0, 105.7], "PURE": [10, 65.2, 23.5, 61.2, 48.6, 68.0, 45.4, 124.4], "UBFC-Phys": [56, 81.7, 13.1, 81.7, 72.0, 93.0, 54.1, 106.2]}
    for dataset, olds in old_table.items():
        if dataset not in table["rows"]:
            continue
        for field, old in zip(["n", "mean", "sd", "median", "q1", "q3", "min", "max"], olds):
            value = table["rows"][dataset][field]
            add("Table7", dataset + "_" + field, old, value, table["rows"][dataset]["n"], "KR full-release BVP with Table7 estimator; external selection and estimator unchanged", value if field == "n" else f"{value:.1f}")
    for field, old in [("welch_p", 0.0625), ("mw_p", 0.001), ("mw_holm", 0.003)]:
        if "PURE" not in table["comparisons"]:
            continue
        add("Table7 comparison", "PURE_" + field, old, table["comparisons"]["PURE"][field], "293 vs10", "new KR Table7 reference; identical external subset")
    return pd.DataFrame(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["release-root", "protocol-dir", "output-dir"]:
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ["pure-root", "mmpd-root", "ubfc-phys-root"]:
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--kr-only", action="store_true", help="Summarize KR data only; omit external Table 7 inputs and comparisons")
    args = parser.parse_args(argv)
    missing = ["--" + name for name in ["pure-root", "mmpd-root", "ubfc-phys-root"]
               if getattr(args, name.replace("-", "_")) is None]
    if not args.kr_only and missing:
        parser.error("required unless --kr-only: " + ", ".join(missing))
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "release_metadata_summary.json").exists() or (out / "session_metrics.csv").exists():
        raise FileExistsError("Summary outputs already exist; use a new output directory")
    inputs = []
    p = read_csv(args.release_root / "metadata/participants.csv", inputs)
    av = read_csv(args.release_root / "metadata/task_session_availability.csv", inputs)
    dictionary = read_csv(args.release_root / "metadata/participants_dictionary.csv", inputs)
    refs = read_csv(args.protocol_dir / "reference_hr.csv", inputs, "protocol")
    eligibility = read_csv(args.protocol_dir / "eligibility_manifest.csv", inputs, "protocol")
    if len(p) != 296 or p.participant_key.duplicated().any() or len(refs) != 1166 or refs.session.duplicated().any():
        raise ValueError("Unexpected or duplicated analysis population")
    released = eligibility[eligibility.released.eq(1)]
    if set(released.session) != set(refs.session) or not released.eligible.eq(1).all():
        raise ValueError("Not all released sessions are represented in the reference table")
    people, eligible = p.set_index("participant_key"), eligibility.set_index("session")
    metadata = make_metadata(p, av, dictionary)
    metrics = []
    for i, (_, ref) in enumerate(refs.iterrows(), 1):
        metrics.append(one_session(args.release_root, ref, people.loc[ref.participant_key], eligible.loc[ref.session], inputs))
        if i % 200 == 0:
            print(f"Summarized {i} release sessions", flush=True)
    m = pd.DataFrame(metrics)
    signal = summarize_signals(m)
    save_json(out / "release_metadata_summary.json", metadata)
    save_json(out / "release_signal_summary.json", signal)
    m.to_csv(out / "session_metrics.csv", index=False)
    dictionary.to_csv(out / "table6_dictionary_verified.csv", index=False)
    print("Release summary complete; summarizing KR Table7 reference" if args.kr_only else
          "Release summary complete; computing external Table7 references", flush=True)
    table = table7(args, refs, out)
    save_json(out / "table7_results.json", table)
    before_after(metadata, signal, table).to_csv(out / "before_after_values.csv", index=False)
    save_json(out / "statistics_provenance.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "script": str(Path(__file__).resolve()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "protocol_path": str(args.protocol_dir / "analysis_protocol.json"),
        "protocol_sha256": digest((args.protocol_dir / "analysis_protocol.json").read_bytes()),
        "reference_provenance_sha256": digest((args.protocol_dir / "reference_hr_provenance.json").read_bytes()),
        "inputs": inputs, "output_sha256": {p.name: digest(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()},
        "scope": ("KR metadata and signals from frozen released files; KR Table7 reference from P1; external inputs and comparisons omitted (--kr-only)" if args.kr_only else
                  "KR metadata and signals only from frozen released files; reference values from P1 release BVP calculation; external Table7 input provenance is separate"),
        "counts": {"participants": len(p), "sessions": len(m), "frame_aligned_samples": int(m.n_frames.sum())},
        "limitations": ["No physiological ground-truth claim is made from BVP equal-to-zero/127 checks", "Camera drop rates are not inferred from historical synchronization NaNs", "P1 AVI header counts require P3 decoded-count reconciliation", "BMI >=100 sensitivity is descriptive and does not alter primary data or cohort"]})
    print(json.dumps({"metadata_bmi": metadata["bmi_primary"], "hr_ge90": signal["benchmark_reference_hr"]["sessions_ge90"], "table7": table["rows"], "table7_comparisons": table["comparisons"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
