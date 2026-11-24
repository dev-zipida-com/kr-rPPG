"""summarize_paired_utility.py: the model list is selectable and the default stays PhysNet + PhysFormer."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "benchmark" / "summarize_paired_utility.py"
EXTERNAL = ("pure", "ubfc_rppg", "mmpd")
FIELDS = ["model", "fold", "session", "participant_key", "task", "pred_hr", "reference_hr", "abs_error"]
# one synthetic participant per fold, two sessions each
SESSIONS = [(f"p{fold + 1:03d}_T0{task}", f"p{fold + 1:03d}", f"T0{task}", str(fold))
            for fold in range(5) for task in range(2)]
EXT_SESSIONS = [f"rec{index:02d}" for index in range(4)]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_result(path: Path, rows: list[dict], reference: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".provenance.json").write_text(json.dumps({
        "partial_monitoring": False, "model_specific_label_hr_used": False, "reference_sha256": sha(reference)}))


def internal_rows(models: list[str], offsets: dict[str, float]) -> list[dict]:
    """Out-of-fold rows: every session once, predicted HR = reference + offset[model]."""
    return [{"model": model, "fold": fold, "session": session, "participant_key": participant, "task": task,
             "pred_hr": 70.0 + offsets[model], "reference_hr": 70.0, "abs_error": abs(offsets[model])}
            for model in models for session, participant, task, fold in SESSIONS]


def external_rows(models: list[str], offsets: dict[str, float]) -> list[dict]:
    """Every fold model is tested on every external recording."""
    return [{"model": model, "fold": str(fold), "session": session, "participant_key": "", "task": "",
             "pred_hr": 80.0 + offsets[model], "reference_hr": 80.0, "abs_error": abs(offsets[model])}
            for model in models for fold in range(5) for session in EXT_SESSIONS]


def build_inputs(
    tmp_path: Path, released_models: list[str], raw_models: list[str]
) -> tuple[Path, Path, dict[str, float], dict[str, float]]:
    reference = tmp_path / "reference_hr.csv"
    with reference.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "participant_key", "task", "fold", "reference_hr"])
        for session, participant, task, fold in SESSIONS:
            writer.writerow([session, participant, task, fold, 70.0])
    external = tmp_path / "external"
    external.mkdir()
    for dataset in EXTERNAL:
        with (external / f"{dataset}_reference_hr.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["session", "reference_hr"])
            for session in EXT_SESSIONS:
                writer.writerow([session, 80.0])
    released = {model: 1.0 + index for index, model in enumerate(released_models)}
    raw = {model: 0.5 + index for index, model in enumerate(raw_models)}
    write_result(tmp_path / "released" / "kr_kfold.csv", internal_rows(released_models, released), reference)
    write_result(tmp_path / "raw" / "kr_kfold.csv", internal_rows(raw_models, raw), reference)
    write_result(tmp_path / "cross" / "raw_to_release.csv",
                 internal_rows(raw_models, {m: v + 0.25 for m, v in raw.items()}), reference)
    write_result(tmp_path / "cross" / "release_to_raw.csv",
                 internal_rows(raw_models, {m: released[m] + 0.75 for m in raw_models}), reference)
    for dataset in EXTERNAL:
        ext_reference = external / f"{dataset}_reference_hr.csv"
        write_result(tmp_path / "released" / f"kr_to_{dataset}.csv",
                     external_rows(released_models, {m: v + 2 for m, v in released.items()}), ext_reference)
        write_result(tmp_path / "raw" / f"kr_to_{dataset}.csv",
                     external_rows(raw_models, {m: v + 3 for m, v in raw.items()}), ext_reference)
    return reference, external, released, raw


def run(tmp_path: Path, reference: Path, external: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--release-results", str(tmp_path / "released"),
         "--raw-results", str(tmp_path / "raw"), "--raw-to-release", str(tmp_path / "cross" / "raw_to_release.csv"),
         "--release-to-raw", str(tmp_path / "cross" / "release_to_raw.csv"), "--reference", str(reference),
         "--external-reference-dir", str(external), "--output", str(tmp_path / "out"), *extra],
        capture_output=True, text=True)


def cells(tmp_path: Path) -> list[dict[str, str]]:
    with (tmp_path / "out" / "table10_cells.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_default_models_are_physnet_and_physformer_with_twenty_cells(tmp_path: Path) -> None:
    released_models = ["physnet", "physformer", "deepphys", "tscan", "efficientphys"]
    reference, external, _, _ = build_inputs(tmp_path, released_models, ["physnet", "physformer"])

    result = run(tmp_path, reference, external)

    assert result.returncode == 0, result.stderr
    rows = cells(tmp_path)
    assert len(rows) == 20
    assert {row["model"] for row in rows} == {"physnet", "physformer"}


def test_requested_models_give_ten_cells_each_with_the_expected_values(tmp_path: Path) -> None:
    released_models = ["physnet", "physformer", "deepphys", "tscan", "efficientphys"]
    control_models = ["deepphys", "tscan", "efficientphys"]
    reference, external, released, raw = build_inputs(tmp_path, released_models, control_models)

    result = run(tmp_path, reference, external, "--models", *control_models)

    assert result.returncode == 0, result.stderr
    rows = cells(tmp_path)
    assert len(rows) == 30
    assert {row["model"] for row in rows} == set(control_models)
    table = {(r["model"], r["training_videos"], r["test_dataset"], r["test_videos"]): float(r["mae"]) for r in rows}
    for model in control_models:
        assert table[(model, "released", "kr_rppg", "released")] == pytest.approx(released[model])
        assert table[(model, "raw", "kr_rppg", "raw")] == pytest.approx(raw[model])
        assert table[(model, "raw", "kr_rppg", "released")] == pytest.approx(raw[model] + 0.25)
        assert table[(model, "released", "kr_rppg", "raw")] == pytest.approx(released[model] + 0.75)
        for dataset in EXTERNAL:
            assert table[(model, "released", dataset, "original_external")] == pytest.approx(released[model] + 2)
            assert table[(model, "raw", dataset, "original_external")] == pytest.approx(raw[model] + 3)
    report = json.loads((tmp_path / "out" / "table10_results.json").read_text())
    assert report["models"] == control_models
    assert len(report["paired_contrasts"]) == 6 * len(control_models)


def test_control_results_must_hold_exactly_the_requested_models(tmp_path: Path) -> None:
    released_models = ["physnet", "physformer", "deepphys", "tscan", "efficientphys"]
    reference, external, _, _ = build_inputs(tmp_path, released_models, ["deepphys", "tscan"])

    result = run(tmp_path, reference, external, "--models", "deepphys", "tscan", "efficientphys")

    assert result.returncode != 0
    assert "control model set" in result.stderr
    assert not (tmp_path / "out" / "table10_cells.csv").exists()


def test_requested_model_missing_from_the_released_results_is_an_error(tmp_path: Path) -> None:
    released_models = ["physnet", "physformer", "deepphys", "tscan", "efficientphys"]
    reference, external, released, _ = build_inputs(tmp_path, released_models, ["deepphys", "tscan"])
    write_result(tmp_path / "released" / "kr_kfold.csv",
                 internal_rows(["physnet", "physformer", "deepphys"], released), reference)

    result = run(tmp_path, reference, external, "--models", "deepphys", "tscan")

    assert result.returncode != 0
    assert "Incomplete OOF cohort" in result.stderr
    assert not (tmp_path / "out" / "table10_cells.csv").exists()


def test_unknown_model_name_is_rejected(tmp_path: Path) -> None:
    reference, external, _, _ = build_inputs(tmp_path, ["physnet", "physformer"], ["physnet", "physformer"])

    result = run(tmp_path, reference, external, "--models", "physnet", "resnet")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
