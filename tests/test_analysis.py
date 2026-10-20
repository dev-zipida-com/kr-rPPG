"""Figure 3 / Table 8 aggregation on synthetic predictions (no real data); Table 7 is tested in test_table7.py."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmark"))


def _module(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


figure3 = _module("figure3_values", "analysis/figure3/figure3_values.py")
table8 = _module("summarize_table8", "benchmark/summarize_table8.py")
MODELS = ("physnet", "physformer", "deepphys", "tscan", "efficientphys")


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    rows = []
    for p in range(901, 941):
        pid = f"p{p}"
        ref = rng.normal(80, 12)
        for task in ("T00", "T01", "T02", "T03"):
            for k, model in enumerate(MODELS):
                label = ref if k >= 2 else ref + rng.normal(0, 0.5)  # 180-frame models share one label
                pred = label + rng.normal(0, 1 + k)
                rows.append({"family": "kr_kfold_eval", "model": model, "fold": str(p % 5), "session": f"{pid}_{task}",
                             "participant_key": pid, "task": task, "pred_hr": pred, "label_hr": label,
                             "abs_error": abs(pred - label)})
    return pd.DataFrame(rows)


def test_figure3_all_group_equals_table8_internal_row(predictions):
    values = figure3.compute(predictions)
    internal = table8.internal_row(predictions)
    for key, name in figure3.MODELS.items():
        assert values[name]["All"]["mae"] == pytest.approx(internal[key])
        cell = values[name]["T01"]
        assert cell["n_sessions"] == 40 and cell["n_participants"] == 40
        assert cell["ci95_lo"] <= cell["mae"] <= cell["ci95_hi"]


def test_bootstrap_does_not_depend_on_row_order(predictions):
    first = figure3.compute(predictions)
    second = figure3.compute(predictions.sample(frac=1, random_state=1))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_kr_to_external_row_is_the_mean_of_fold_maes():
    df = pd.DataFrame({"model": ["physnet"] * 4, "fold": ["0", "0", "1", "1"], "abs_error": [1.0, 3.0, 5.0, 5.0]})
    assert table8.kr_to_external_row(df)["physnet"] == pytest.approx(3.5)


def test_rows_without_manuscript_values_print_a_dash(tmp_path, capsys):
    expected = {family: values for family, _label, values in table8.ROWS}
    assert expected["mmpd_to_kr_eval"] is None  # no reference values listed for MMPD -> KR-rPPG
    pd.DataFrame({"model": list(MODELS), "fold": [""] * 5, "session": ["p901_T01"] * 5,
                  "participant_key": ["p901"] * 5, "abs_error": [1.0, 2.0, 3.0, 4.0, 5.0]}).to_csv(
        tmp_path / "mmpd_to_kr_eval.csv", index=False)
    assert table8.main(["--results", str(tmp_path)]) == 0
    line = next(text for text in capsys.readouterr().out.splitlines() if "MMPD -> KR-rPPG" in text)
    assert "1.00" in line and "5.00" in line and "(    -)" in line


def test_expected_figure3_values_match_table8_manuscript_row():
    expected = json.loads((REPO / "analysis" / "figure3" / "expected_values.json").read_text())["values"]
    row = [expected[name]["All"]["mae"] for name in ("PhysNet", "PhysFormer", "DeepPhys", "TS-CAN", "EfficientPhys")]
    assert row == [1.47, 1.51, 11.83, 7.48, 4.81]
    assert table8.ROWS[0][2] == tuple(row)


def test_readme_table8_lists_the_reference_values_of_summarize_table8():
    section = (REPO / "README.md").read_text().split("## Expected results (manuscript)", 1)[1].split("\n## ", 1)[0]
    readme = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 6 and "->" in cells[0] and all(c.replace(".", "", 1).isdigit() for c in cells[1:]):
            label = cells[0].replace(" (5-fold, internal)", "")
            readme[label] = tuple(float(cell) for cell in cells[1:])
    listed = {label.split(None, 1)[1]: values for _family, label, values in table8.ROWS if values is not None}
    assert len(readme) == 7 and readme == listed
