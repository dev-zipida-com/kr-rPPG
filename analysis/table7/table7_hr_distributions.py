#!/usr/bin/env python3
"""Table 7: stationary-rest reference heart-rate distributions, KR-rPPG versus three public rPPG datasets.

    python analysis/table7/table7_hr_distributions.py --kr $KR_RPPG_WORK/results/kr_kfold_eval.csv \
        --pure $PURE_ROOT --mmpd $MMPD_ROOT --ubfc-phys $UBFC_PHYS_ROOT

Without an option the script uses $KR_RPPG_WORK/results/kr_kfold_eval.csv, $PURE_ROOT, $MMPD_ROOT and
$UBFC_PHYS_ROOT if these environment variables are set; a row whose input is not given is skipped. The dataset
roots are the raw folders also used by the benchmark (rPPG-Toolbox layouts); nothing is written.

Rows (selection of the Table 7 caption), one heart rate (HR) per recording:
  KR-rPPG    `label_hr` of the DeepPhys rows of task T01 (stationary) in the KR-rPPG 5-fold prediction table
             written by benchmark/collect_predictions.py: the rPPG-Toolbox reference HR of the contact BVP over the
             whole session (shared by the three 180-frame models; the reference HR of Figure 3). N = 293.
  PURE       steady task (`<root>/NN-01/NN-01.json`): pulse-oximeter waveform `/FullPackage[*].Value.waveform`,
             sampling rate fs = (n - 1) / duration from the `Timestamp` field (ns). N = 10.
  MMPD       `<root>/subjectN/*.mat`: `GT_ppg` at 30 Hz of the recordings whose `motion` field is exactly
             'Stationary' (this excludes 'Stationary (after exercise)'); the HRs of a subject's recordings are
             averaged (mean), one value per subject. N = 33.
  UBFC-Phys  resting task T1 (`<root>/sN/bvp_sN_T1.csv`): Empatica E4 BVP at 64 Hz (first column, no header), all
             subjects. N = 56.

Reference HR of the public datasets, from the raw reference waveform over the whole recording:
  order-1 Butterworth band-pass 0.75-2.5 Hz (scipy.signal.butter), zero-phase filtering (filtfilt), no detrending;
  periodogram with scipy.signal.periodogram(fs, nfft = next power of two >= length, detrend=False);
  HR = 60 x the frequency of the largest power within 0.75-2.5 Hz (45-150 bpm).

Statistics: N, mean, SD (ddof = 1), median and IQR (numpy percentiles, linear interpolation), range; printed with
one decimal. KR-rPPG versus each public dataset: Welch t-test and Mann-Whitney U (both two-sided, scipy defaults),
Holm-adjusted across the three comparisons within each test.

Every printed value is compared with the manuscript (EXPECTED: Table 7; MANUSCRIPT_STATEMENTS: the statistics
reported with it) and with the p values of the original analysis (EXPECTED_P, three significant digits), and
flagged 'match' or 'MISMATCH'. Output: aggregates only (no per-recording or per-participant values).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt, periodogram

BAND_HZ = (0.75, 2.5)  # 45-150 bpm
FILTER_ORDER = 1
FS_MMPD = 30.0
FS_UBFC_PHYS = 64.0
KR_MODEL, KR_TASK = "deepphys", "T01"
MMPD_MOTION = "Stationary"
EXTERNAL = ("MMPD", "PURE", "UBFC-Phys")  # order of the Table 7 rows and of the Holm families

EXPECTED = {  # manuscript Table 7: N, mean, SD, median, Q1, Q3, min, max
    "KR-rPPG": (293, 81.0, 13.8, 80.4, 70.8, 90.1, 52.7, 130.1),
    "MMPD": (33, 76.8, 11.7, 75.6, 70.5, 82.8, 51.0, 105.7),
    "PURE": (10, 65.2, 23.5, 61.2, 48.6, 68.0, 45.4, 124.4),
    "UBFC-Phys": (56, 81.7, 13.1, 81.7, 72.0, 93.0, 54.1, 106.2),
}
EXPECTED_P = {  # original analysis, three significant digits: (Welch p, Mann-Whitney p), KR-rPPG versus dataset
    "MMPD": (0.0600, 0.106),
    "PURE": (0.0625, 0.00104),
    "UBFC-Phys": (0.724, 0.602),
}
MANUSCRIPT_STATEMENTS = (
    "Welch t-test: all three two-sided p >= 0.060",
    "Mann-Whitney U, KR-rPPG vs PURE: p = 0.001 (Holm-adjusted 0.003)",
    "Mann-Whitney U, KR-rPPG vs MMPD and UBFC-Phys: not significant after Holm (>= 0.05)",
)
ALPHA = 0.05

Description = Tuple[float, ...]  # N, mean, SD, median, Q1, Q3, min, max


# ----------------------------------------------------------------------------- reference heart rate
def next_pow2(n: int) -> int:
    """Smallest power of two >= n (n >= 1)."""
    if n < 1:
        raise ValueError(f"length must be >= 1, got {n}")
    return 1 << (int(n) - 1).bit_length()


def reference_hr(signal: Sequence[float], fs: float, band: Tuple[float, float] = BAND_HZ) -> float:
    """HR (bpm) of a whole reference waveform: band-pass, zero-padded periodogram, peak frequency in `band` x 60."""
    x = np.asarray(signal, dtype=float).ravel()
    if not np.all(np.isfinite(x)):
        raise ValueError("the reference waveform contains non-finite samples")
    if not fs > 2 * band[1]:
        raise ValueError(f"sampling rate {fs} Hz is too low for the {band[0]}-{band[1]} Hz band")
    b, a = butter(FILTER_ORDER, [band[0] / fs * 2, band[1] / fs * 2], btype="bandpass")
    pad = 3 * max(len(a), len(b))  # filtfilt default padding
    if len(x) <= pad:
        raise ValueError(f"the reference waveform has {len(x)} samples; at least {pad + 1} are needed")
    x = filtfilt(b, a, x)
    freqs, power = periodogram(x, fs=fs, nfft=next_pow2(len(x)), detrend=False)
    in_band = (freqs >= band[0]) & (freqs <= band[1])
    return float(freqs[in_band][np.argmax(power[in_band])] * 60.0)


# ----------------------------------------------------------------------------- rows
def kr_values(path: Path) -> np.ndarray:
    """label_hr of the DeepPhys T01 rows of a collect_predictions.py table (one row per participant)."""
    df = pd.read_csv(path, dtype={"model": str, "task": str, "participant_key": str})
    missing = {"model", "task", "label_hr"} - set(df.columns)
    if missing:
        sys.exit(f"{path}: missing column(s) {sorted(missing)}")
    rows = df[(df.model == KR_MODEL) & (df.task == KR_TASK)]
    if rows.empty:
        sys.exit(f"{path}: no {KR_MODEL} rows for task {KR_TASK}")
    if "participant_key" in rows.columns and rows.participant_key.duplicated().any():
        sys.exit(f"{path}: {int(rows.participant_key.duplicated().sum())} duplicate {KR_MODEL} {KR_TASK} rows; "
                 "give the table of one 5-fold evaluation")
    values = rows.label_hr.to_numpy(dtype=float)
    if np.isnan(values).any():
        print(f"KR-rPPG: {int(np.isnan(values).sum())} rows without label_hr ignored", file=sys.stderr)
    return values[~np.isnan(values)]


def pure_recording_hr(path: Path) -> float:
    """HR of one PURE recording; fs from the timestamps (ns) of the pulse-oximeter packages."""
    with open(path, encoding="utf-8") as handle:
        packages = json.load(handle)["/FullPackage"]
    seconds = np.array([p["Timestamp"] for p in packages], dtype=float) / 1e9
    fs = (len(seconds) - 1) / (seconds[-1] - seconds[0])
    return reference_hr([p["Value"]["waveform"] for p in packages], fs)


def pure_values(root: Path) -> np.ndarray:
    """Steady task (session NN-01) of every PURE subject."""
    files = sorted(f for f in root.glob("*-01/*-01.json")
                   if re.fullmatch(r"\d{2}-01", f.stem) and f.parent.name == f.stem)
    if not files:
        sys.exit(f"{root}: no PURE steady recordings (NN-01/NN-01.json)")
    return np.array([pure_recording_hr(f) for f in files])


def _mat_string(value: np.ndarray) -> str:
    return str(np.asarray(value).ravel()[0])


def mmpd_values(root: Path) -> np.ndarray:
    """Per-subject mean HR of the MMPD recordings whose motion is exactly 'Stationary'."""
    import scipy.io as sio

    per_subject: Dict[int, List[float]] = {}
    n_files = 0
    for path in sorted(root.glob("subject*/*.mat")):
        match = re.fullmatch(r"subject(\d+)", path.parent.name)
        if not match:
            continue
        n_files += 1
        mat = sio.loadmat(str(path), variable_names=["GT_ppg", "motion"])
        if _mat_string(mat["motion"]).strip() != MMPD_MOTION:
            continue
        try:
            hr = reference_hr(mat["GT_ppg"], FS_MMPD)
        except ValueError as error:
            sys.exit(f"{path}: {error}")
        per_subject.setdefault(int(match.group(1)), []).append(hr)
    if not per_subject:
        sys.exit(f"{root}: no MMPD recordings with motion '{MMPD_MOTION}' ({n_files} .mat files read)")
    n_recordings = sum(len(v) for v in per_subject.values())
    print(f"MMPD: {n_recordings} '{MMPD_MOTION}' recordings of {len(per_subject)} subjects ({n_files} files)",
          file=sys.stderr)
    return np.array([np.mean(per_subject[s]) for s in sorted(per_subject)])


def ubfc_phys_values(root: Path) -> np.ndarray:
    """Resting task T1 of every UBFC-Phys subject (Empatica E4 BVP)."""
    folders = sorted((d for d in root.glob("s*") if d.is_dir() and re.fullmatch(r"s\d+", d.name)),
                     key=lambda d: int(d.name[1:]))
    values, missing = [], 0
    for folder in folders:
        path = folder / f"bvp_{folder.name}_T1.csv"
        if not path.exists():
            missing += 1
            continue
        bvp = pd.read_csv(path, header=None).iloc[:, 0].astype(float).to_numpy()
        values.append(reference_hr(bvp, FS_UBFC_PHYS))
    if missing:
        print(f"UBFC-Phys: {missing} subject folder(s) without a T1 BVP file", file=sys.stderr)
    if not values:
        sys.exit(f"{root}: no UBFC-Phys T1 BVP files (sN/bvp_sN_T1.csv)")
    return np.array(values)


# ----------------------------------------------------------------------------- descriptive statistics
def describe(x: Sequence[float]) -> Description:
    x = np.asarray(x, dtype=float)
    q1, median, q3 = np.percentile(x, [25, 50, 75])  # numpy default: linear interpolation
    return (len(x), float(x.mean()), float(x.std(ddof=1)), float(median), float(q1), float(q3),
            float(x.min()), float(x.max()))


def cells(d: Description) -> List[str]:
    """The eight Table 7 cells as printed: N, then one decimal."""
    return [str(int(d[0]))] + [f"{v:.1f}" for v in d[1:]]


def format_row(c: Sequence[str]) -> str:
    mean_sd, median_iqr, value_range = f"{c[1]} +/- {c[2]}", f"{c[3]} ({c[4]}-{c[5]})", f"{c[6]}-{c[7]}"
    return f"{c[0]:>4}  {mean_sd:<13}  {median_iqr:<17}  {value_range:<11}"


def matching_cells(d: Description, expected: Description) -> int:
    return sum(a == b for a, b in zip(cells(d), cells(expected)))


# ----------------------------------------------------------------------------- tests
def holm(p: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p values (same order as `p`)."""
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def compare_with_kr(kr: np.ndarray, others: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Welch t-test and Mann-Whitney U (two-sided) of KR-rPPG versus each dataset, Holm within each test."""
    names = list(others)
    welch = [float(stats.ttest_ind(kr, others[n], equal_var=False).pvalue) for n in names]
    mwu = [float(stats.mannwhitneyu(kr, others[n], alternative="two-sided").pvalue) for n in names]
    return {n: {"welch_p": w, "welch_holm": wh, "mw_p": m, "mw_holm": mh}
            for n, w, wh, m, mh in zip(names, welch, holm(welch), mwu, holm(mwu))}


def sig3(p: float) -> str:
    return f"{p:#.3g}"


def manuscript_checks(results: Dict[str, Dict[str, float]]) -> List[Tuple[str, bool]]:
    """MANUSCRIPT_STATEMENTS evaluated on `results` (needs all three datasets)."""
    pure = results["PURE"]
    others = [results[n] for n in EXTERNAL if n != "PURE"]
    return [
        (MANUSCRIPT_STATEMENTS[0], all(round(results[n]["welch_p"], 3) >= 0.060 for n in EXTERNAL)),
        (MANUSCRIPT_STATEMENTS[1], f"{pure['mw_p']:.3f}" == "0.001" and f"{pure['mw_holm']:.3f}" == "0.003"),
        (MANUSCRIPT_STATEMENTS[2], all(r["mw_holm"] >= ALPHA for r in others)),
    ]


# ----------------------------------------------------------------------------- output
def flag(ok: bool) -> str:
    return "match" if ok else "MISMATCH"


def print_rows(data: Dict[str, np.ndarray]) -> int:
    mismatches = 0
    print(f"{'Dataset':10s} {'N':>4}  {'Mean +/- SD':<13}  {'Median (IQR)':<17}  {'Range':<11}  vs manuscript")
    for name, values in data.items():
        d = describe(values)
        n_match = matching_cells(d, EXPECTED[name])
        print(f"{name:10s} {format_row(cells(d))}  {flag(n_match == 8)} ({n_match}/8 cells)")
        if n_match < 8:
            mismatches += 1
            print(f"{'  expected':10s} {format_row(cells(EXPECTED[name]))}")
    return mismatches


def print_tests(results: Dict[str, Dict[str, float]]) -> int:
    mismatches = 0
    print(f"\n{'KR-rPPG vs':10s}  {'Welch p (Holm)':<17}  {'Mann-Whitney p (Holm)':<22}  "
          "vs original analysis (3 significant digits)")
    for name, r in results.items():
        ok = (sig3(r["welch_p"]), sig3(r["mw_p"])) == tuple(sig3(e) for e in EXPECTED_P[name])
        mismatches += not ok
        welch = f"{sig3(r['welch_p'])} ({sig3(r['welch_holm'])})"
        mann_whitney = f"{sig3(r['mw_p'])} ({sig3(r['mw_holm'])})"
        print(f"{name:10s}  {welch:<17}  {mann_whitney:<22}  {flag(ok)}")
    if set(results) != set(EXTERNAL):
        print("(the manuscript statements need all three public datasets)")
        return mismatches
    print("Manuscript statements:")
    for text, ok in manuscript_checks(results):
        mismatches += not ok
        print(f"  {text}: {flag(ok)}")
    return mismatches


def _env_path(variable: str, *parts: str) -> Optional[Path]:
    value = os.environ.get(variable)
    return Path(value, *parts) if value else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kr", type=Path, default=_env_path("KR_RPPG_WORK", "results", "kr_kfold_eval.csv"),
                        help="KR-rPPG 5-fold prediction table (default: $KR_RPPG_WORK/results/kr_kfold_eval.csv)")
    parser.add_argument("--pure", type=Path, default=_env_path("PURE_ROOT"),
                        help="PURE folder with NN-MM/NN-MM.json (default: $PURE_ROOT)")
    parser.add_argument("--mmpd", type=Path, default=_env_path("MMPD_ROOT"),
                        help="MMPD folder with subjectN/*.mat (default: $MMPD_ROOT)")
    parser.add_argument("--ubfc-phys", type=Path, default=_env_path("UBFC_PHYS_ROOT"),
                        help="UBFC-Phys folder with sN/bvp_sN_T1.csv (default: $UBFC_PHYS_ROOT)")
    args = parser.parse_args(argv)

    loaders: List[Tuple[str, Optional[Path], Callable[[Path], np.ndarray]]] = [
        ("KR-rPPG", args.kr, kr_values), ("MMPD", args.mmpd, mmpd_values), ("PURE", args.pure, pure_values),
        ("UBFC-Phys", args.ubfc_phys, ubfc_phys_values)]
    data: Dict[str, np.ndarray] = {}
    for name, path, load in loaders:
        if path is None:
            continue
        if not path.exists():
            sys.exit(f"{name}: {path} does not exist")
        data[name] = load(path)
    if not data:
        parser.error("give at least one of --kr, --pure, --mmpd, --ubfc-phys")

    mismatches = print_rows(data)
    others = {name: data[name] for name in EXTERNAL if name in data}
    if "KR-rPPG" in data and others:
        mismatches += print_tests(compare_with_kr(data["KR-rPPG"], others))
    summary = "all compared values match" if mismatches == 0 else f"{mismatches} MISMATCH(ES)"
    print(f"\n{summary} (manuscript Table 7 and its statistics)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
