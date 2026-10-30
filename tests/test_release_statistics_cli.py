import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "release_statistics", REPO / "analysis/release_statistics/summarize_release.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)
BASE_ARGS = ["--release-root", "release", "--protocol-dir", "protocol", "--output-dir", "out"]


def test_external_roots_remain_required_by_default(capsys):
    with pytest.raises(SystemExit) as stopped:
        summary.parse_args(BASE_ARGS)
    assert stopped.value.code == 2
    assert "required unless --kr-only" in capsys.readouterr().err
    args = summary.parse_args(BASE_ARGS + ["--pure-root", "pure", "--mmpd-root", "mmpd",
                                          "--ubfc-phys-root", "ubfc"])
    assert args.kr_only is False


def test_kr_only_does_not_read_external_data(tmp_path):
    args = summary.parse_args(BASE_ARGS + ["--kr-only"])

    class ForbiddenExternalRoot:
        def glob(self, pattern):
            pytest.fail("KR-only mode attempted to read external data")

    args.pure_root = args.mmpd_root = args.ubfc_phys_root = ForbiddenExternalRoot()
    reference = pd.DataFrame([
        {"participant_key": f"p{i:03d}", "task": "T01", "table7_reference_hr": 75.0}
        for i in range(1, 294)
    ] + [{"participant_key": "p001", "task": "T00", "table7_reference_hr": 120.0}])
    result = summary.table7(args, reference, tmp_path)
    assert result["scope"] == "kr_only"
    assert list(result["rows"]) == ["KR-rPPG"]
    assert result["rows"]["KR-rPPG"]["n"] == 293
    assert result["rows"]["KR-rPPG"]["mean"] == 75.0
    assert result["rows"]["KR-rPPG"]["sd"] == 0.0
    assert result["comparisons"] == {}
    assert len(pd.read_csv(tmp_path / "table7_participant_values.csv")) == 293
    assert json.loads((tmp_path / "table7_input_manifest.json").read_text())["external_inputs"] == []
    assert not (tmp_path / "table7_external_recordings.csv").exists()
