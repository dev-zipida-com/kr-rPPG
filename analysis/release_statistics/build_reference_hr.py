#!/usr/bin/env python3
"""Build release-only scalar HR references and the common evaluation manifest.

Inputs are the released metadata, synchronized BVP, timestamps, and AVI headers.
No original-study prediction table or private participant mapping is read.

Benchmark: skip 180 samples; N=min(BVP rows, AVI header frames)-180;
take min(floor(N/L)*L for L in (128,160,180)) samples. Apply sparse
smoothness-prior detrending (lambda=100), order-1 Butterworth 0.6--3.3 Hz
with filtfilt, and a next-power-of-two periodogram (detrend=False).
The scalar reference is 60 times the strongest in-band frequency at 30 Hz.

Table 7: for T01 only, use the complete released BVP, no initial skip,
no detrending, order-1 Butterworth 0.75--2.5 Hz with filtfilt and the same
periodogram/peak rule. This is a separately named descriptive endpoint.

AVI header counts establish initial eligibility, not successful decode counts.
The cache builder must reconcile its decoded counts before final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import scipy
from scipy.signal import butter, filtfilt, periodogram
from scipy.sparse import eye, spdiags
from scipy.sparse.linalg import spsolve

FS = 30.0
SKIP = 180
CHUNKS = (128, 160, 180)
BENCHMARK_BAND = (0.6, 3.3)
TABLE7_BAND = (0.75, 2.5)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def detrend(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    n = len(x)
    ones = np.ones(n)
    d = spdiags(np.array([ones, -2 * ones, ones]), [0, 1, 2], n - 2, n)
    a = eye(n) + lam**2 * d.T.dot(d)
    return x - spsolve(a.tocsc(), x)


def scalar_hr(x: np.ndarray, band: tuple[float, float], smooth: bool) -> float:
    if not np.isfinite(x).all():
        raise ValueError("Reference BVP contains nonfinite values")
    if len(x) < 10:
        raise ValueError("Reference BVP has fewer than 10 samples")
    signal = detrend(x) if smooth else x
    b, a = butter(1, [v * 2 / FS for v in band], btype="bandpass")
    signal = filtfilt(b, a, signal)
    frequencies, power = periodogram(
        signal, fs=FS, nfft=1 << (len(signal) - 1).bit_length(), detrend=False
    )
    selected = (frequencies >= band[0]) & (frequencies <= band[1])
    return float(frequencies[selected][np.argmax(power[selected])] * 60)


def video_header(path: Path) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open released video: {path}")
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(n) or n < 1 or abs(n - round(n)) > 1e-6:
            raise ValueError(f"Invalid AVI header frame count {n}: {path}")
        return int(round(n)), float(fps)
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.release_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = [out / n for n in ("reference_hr.csv", "eligibility_manifest.csv", "reference_hr_provenance.json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Reference outputs already exist; choose a new output directory")

    metadata_names = ("participants.csv", "task_session_availability.csv", "fold_assignment.csv")
    metadata = {n: read_csv(root / "metadata" / n) for n in metadata_names}
    participants = {r["participant_key"] for r in metadata["participants.csv"]}
    availability = metadata["task_session_availability.csv"]
    folds = {r["participant_key"]: int(r["fold"]) for r in metadata["fold_assignment.csv"]}
    if len(participants) != len(metadata["participants.csv"]):
        raise ValueError("Duplicate participant keys")
    if len(folds) != len(metadata["fold_assignment.csv"]):
        raise ValueError("Duplicate fold assignment")
    if len(availability) != len(participants) or {r["participant_key"] for r in availability} != participants or set(folds) != participants:
        raise ValueError("Metadata participant sets differ")

    eligibility, references, inputs = [], [], []
    for participant in sorted(availability, key=lambda r: r["participant_key"]):
        key = participant["participant_key"]
        for task_i in range(4):
            task = f"T0{task_i}"
            flag = participant[f"t0{task_i}_released"]
            if flag not in ("0", "1"):
                raise ValueError(f"Invalid release flag for {key}/{task}: {flag}")
            released = int(flag)
            row = {
                "session": f"{key}_{task}", "session_id": f"{key}/{task}",
                "participant_key": key, "task": task, "fold": folds[key],
                "released": released, "bvp_rows": "", "timestamp_rows": "",
                "video_frames_header": "", "video_header_fps": "", "bbox_rows": "",
                "skip_frames": SKIP, "available_frames_after_skip": "",
                "model128_frames": "", "model160_frames": "", "model180_frames": "",
                "common_eval_frames": "", "eval_start_frame": "", "eval_end_frame_exclusive": "",
                "eligible": 0, "reason": "not_released",
                "frame_count_status": "not_applicable",
                "bbox_limits_evaluation": 0,
            }
            if not released:
                eligibility.append(row)
                continue

            directory = root / key / task
            bvp_bytes = (directory / "BVP_synchronized.csv").read_bytes()
            bvp_rows = list(csv.DictReader(io.StringIO(bvp_bytes.decode("utf-8-sig"))))
            bvp = np.array([float(r["bvp"]) for r in bvp_rows], dtype=np.float64)
            timestamps = read_csv(directory / "frames_timestamp.csv")
            bbox_rows = len(read_csv(directory / "face_bboxes.csv"))
            if len(bvp) != len(timestamps):
                raise ValueError(f"BVP/timestamp row mismatch: {key}/{task}")
            if not np.isfinite(bvp).all():
                raise ValueError(f"Nonfinite BVP values: {key}/{task}")
            video_frames, video_fps = video_header(directory / "video_bfdt_ffv1.avi")
            n = max(0, min(len(bvp), video_frames) - SKIP)
            model_frames = {length: (n // length) * length for length in CHUNKS}
            common = min(model_frames.values())
            eligible = common > 0
            row.update({
                "bvp_rows": len(bvp), "timestamp_rows": len(timestamps),
                "video_frames_header": video_frames, "video_header_fps": video_fps,
                "bbox_rows": bbox_rows, "available_frames_after_skip": n,
                **{f"model{length}_frames": count for length, count in model_frames.items()},
                "common_eval_frames": common, "eval_start_frame": SKIP,
                "eval_end_frame_exclusive": SKIP + common,
                "eligible": int(eligible), "reason": "eligible" if eligible else "insufficient_frames_after_skip",
                "frame_count_status": "avi_header_pending_decode_reconciliation",
            })
            eligibility.append(row)
            selected = bvp[SKIP:SKIP + common]
            if eligible:
                if len(selected) != common:
                    raise ValueError(f"Reference window does not match manifest: {key}/{task}")
                references.append({
                    "session": row["session"], "session_id": row["session_id"],
                    "participant_key": key, "task": task, "fold": folds[key],
                    "reference_hr": scalar_hr(selected, BENCHMARK_BAND, True),
                    "table7_reference_hr": scalar_hr(bvp, TABLE7_BAND, False) if task == "T01" else "",
                    "common_eval_frames": common, "eval_start_frame": SKIP,
                    "eval_end_frame_exclusive": SKIP + common,
                    "sampling_rate_hz": FS,
                    "bvp_file_sha256": sha256(bvp_bytes),
                    "benchmark_bvp_float64_le_sha256": sha256(selected.astype("<f8").tobytes()),
                })
            inputs.append({
                "session": row["session"], "bvp_sha256": sha256(bvp_bytes),
                "video_header_frames": video_frames, "bvp_rows": len(bvp),
                "bbox_rows": bbox_rows,
            })
            if len(inputs) % 200 == 0:
                print(f"Processed {len(inputs)} released sessions", flush=True)

    if not references:
        raise ValueError("No eligible reference sessions")
    if len({r["session"] for r in eligibility}) != len(eligibility):
        raise ValueError("Duplicate session keys")
    write_csv(outputs[0], references)
    write_csv(outputs[1], eligibility)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "release_root": str(root), "script_sha256": sha256(Path(__file__).read_bytes()),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "opencv": cv2.__version__},
        "metadata_sha256": {n: sha256((root / "metadata" / n).read_bytes()) for n in metadata_names},
        "protocol": {
            "nominal_sampling_rate_hz": FS, "skip_frames": SKIP,
            "model_chunk_lengths": {"physnet": 128, "physformer": 160, "deepphys": 180, "tscan": 180, "efficientphys": 180},
            "available_frames": "max(0, min(BVP rows, AVI header frame count) - 180)",
            "common_eval_frames": "min(floor(N/128)*128, floor(N/160)*160, floor(N/180)*180)",
            "benchmark": "Raw released BVP[180:180+common_eval_frames]; sparse smoothness-prior detrend lambda=100; order-1 Butterworth 0.6-3.3 Hz; filtfilt; periodogram nfft=next power of two, detrend=False; strongest in-band frequency times 60; scalar bpm",
            "table7": "T01 full released BVP; no skip and no detrend; order-1 Butterworth 0.75-2.5 Hz; filtfilt; periodogram nfft=next power of two, detrend=False; strongest in-band frequency times 60; scalar bpm",
            "bbox": "Recorded for annotation coverage only; never limits the evaluation interval",
            "predictions": "Each model's concatenated prediction must be trimmed to common_eval_frames before scalar HR estimation; same scalar reference_hr for every model",
            "decode_gate": "Cache builder must verify decoded frame counts and emitted evaluation intervals against this manifest; any disagreement requires reconciled manifests and references before final metrics",
        },
        "counts": {
            "participants": len(participants), "planned_slots": len(eligibility),
            "released_sessions": len(inputs), "eligible_sessions": len(references),
            "table7_sessions": sum(r["task"] == "T01" for r in references),
            "reasons": dict(Counter(r["reason"] for r in eligibility)),
            "task_counts": dict(Counter(r["task"] for r in references)),
            "header_bvp_count_mismatches": sum(r["video_header_frames"] != r["bvp_rows"] for r in inputs),
            "short_bbox_sessions": sum(r["bbox_rows"] < r["bvp_rows"] for r in inputs),
        },
        "outputs_sha256": {p.name: sha256(p.read_bytes()) for p in outputs[:2]},
        "session_inputs": inputs,
    }
    outputs[2].write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance["counts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
