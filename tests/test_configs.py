"""The committed benchmark configs are exactly what benchmark/make_configs.py generates, use only known
${PLACEHOLDER}s, contain no absolute paths, and load through the toolbox config system."""

import csv
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "benchmark" / "configs"
sys.path.insert(0, str(REPO / "benchmark"))

import make_configs  # noqa: E402

PLACEHOLDERS = {"KR_RPPG_ROOT", "KR_RPPG_WORK", "KR_RPPG_CHECKPOINTS", "RPPG_TOOLBOX_WEIGHTS",
                "PURE_ROOT", "UBFC_RPPG_ROOT", "MMPD_ROOT", "UBFC_PHYS_ROOT"}


def test_committed_configs_are_up_to_date():
    assert make_configs.main(["--check"]) == 0


def test_config_counts():
    counts = {}
    for path in CONFIGS.rglob("*.yaml"):
        family = path.relative_to(CONFIGS).parts[0]
        counts[family] = counts.get(family, 0) + 1
    assert counts == {"preprocess": 25, "kr_kfold": 50, "kr_to_pure": 25, "kr_to_ubfc_rppg": 25, "kr_to_mmpd": 25,
                      "kr_to_ubfc_phys": 25, "pure_to_kr": 5, "ubfc_rppg_to_kr": 5, "mmpd_to_kr": 10}


@pytest.mark.parametrize("path", sorted(CONFIGS.rglob("*.yaml")), ids=lambda p: str(p.relative_to(CONFIGS)))
def test_only_known_placeholders_and_no_absolute_paths(path):
    text = path.read_text()
    assert set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", text)) <= PLACEHOLDERS
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        value = line.split(":", 1)[1].strip() if ":" in line else line.strip().lstrip("- ").strip()
        assert not value.startswith("/"), line


def test_checkpoint_paths_match_the_manifest():
    """Official weights use the upstream manifest; local models use fixed family aliases."""
    with open(REPO / "checkpoints" / "checkpoints_manifest.csv", newline="") as handle:
        files = {row["file"] for row in csv.DictReader(handle)}
    upstream, trained_by_users = set(), set()
    for path in CONFIGS.rglob("*.yaml"):
        model_path = yaml.safe_load(path.read_text())["INFERENCE"]["MODEL_PATH"]
        if model_path.startswith("${KR_RPPG_CHECKPOINTS}/"):
            trained_by_users.add(model_path.split("/", 1)[1])
        elif model_path.startswith("${RPPG_TOOLBOX_WEIGHTS}/"):
            upstream.add("upstream:final_model_release/" + model_path.split("/", 1)[1])
    assert upstream == files and len(upstream) == 10
    assert len(trained_by_users) == 25 + 5
    assert {name.split("/", 1)[0] for name in trained_by_users} == {"kr_5fold", "mmpd_trained"}
    kr_paths = {name for name in trained_by_users if name.startswith("kr_5fold/")}
    assert len(kr_paths) == 25
    assert all(name.endswith("_Best.pth") for name in kr_paths)


def test_run_stops_before_any_physformer_config_without_the_upstream_loss(tmp_path, monkeypatch, capsys):
    import common
    import run

    sys.path.insert(0, str(REPO / "third_party" / "rPPG-Toolbox"))
    from neural_methods.loss import PhysFormerLossAdapter as adapter

    monkeypatch.delenv(adapter.LOCATION_ENV, raising=False)
    monkeypatch.setattr(adapter, "UPSTREAM_FILE", tmp_path / "PhysFormerLossComputer.py")
    monkeypatch.setenv("KR_RPPG_WORK", str(tmp_path / "work"))
    monkeypatch.setattr(run.subprocess, "run", lambda *args, **kwargs: pytest.fail("a config was started"))
    assert len(run.physformer_configs(common.expand_configs(["kr_kfold/train"]))) == 5
    assert run.physformer_configs(common.expand_configs(["kr_kfold/train/fold0_physnet.yaml"])) == []
    assert run.main(["kr_kfold/train"]) == 1
    assert "tools/fetch_physformer_loss.py" in capsys.readouterr().err
    assert run.main(["kr_kfold/train", "--dry-run"]) == 0  # listing does not need the file
    assert not (tmp_path / "work").exists()


def test_toolbox_loads_a_config_and_refuses_unset_placeholders(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / "third_party" / "rPPG-Toolbox"))
    from config import get_config

    path = str(CONFIGS / "kr_kfold" / "eval" / "fold1_physnet.yaml")
    for name in PLACEHOLDERS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="not set"):
        get_config(Namespace(config_file=path))
    for name in PLACEHOLDERS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    cfg = get_config(Namespace(config_file=path))
    assert cfg.TEST.DATA.KFOLD_ID == 1 and cfg.TEST.DATA.KFOLD_TEST_FOLD
    assert cfg.TEST.DATA.DATA_PATH == os.environ["KR_RPPG_ROOT"]
    assert cfg.INFERENCE.MODEL_PATH.endswith("kr_5fold/physnet/KFold_physnet_fold1_Best.pth")
    assert cfg.MODEL.PHYSFORMER.GRA_SHARP < 0  # default: use THETA, as in KR-rPPG k-fold training


# PhysFormer is scored with the gra_sharp it was trained with: the official rPPG-Toolbox PURE / UBFC-rPPG weights
# were trained by the upstream trainer (2.0); the KR-rPPG fold models and the MMPD models are trained and scored
# with the toolbox default (GRA_SHARP < 0, i.e. THETA = 0.7).
OFFICIAL_PHYSFORMER = ("pure_to_kr/physformer.yaml", "ubfc_rppg_to_kr/physformer.yaml")


def _physformer_gra_sharp():
    """GRA_SHARP as written in each committed PhysFormer config (None: key absent, i.e. the toolbox default)."""
    out = {}
    for path in sorted(CONFIGS.rglob("*.yaml")):
        model = yaml.safe_load(path.read_text())["MODEL"]
        if model["NAME"] == "PhysFormer":
            out[path.relative_to(CONFIGS).as_posix()] = model["PHYSFORMER"].get("GRA_SHARP")
    return out


def test_physformer_configs_use_the_training_gra_sharp():
    values = _physformer_gra_sharp()
    assert {rel: values.get(rel) for rel in OFFICIAL_PHYSFORMER} == dict.fromkeys(OFFICIAL_PHYSFORMER, 2.0)
    kr_models = [rel for rel in values if rel.startswith(("kr_kfold/", "kr_to_"))]
    assert len(kr_models) == 10 + 4 * 5  # kr_kfold train + eval, 4 x kr_to_* (5 folds each)
    assert sum(rel.startswith("kr_to_ubfc_phys/") for rel in kr_models) == 5
    defaults = {rel: value for rel, value in values.items() if rel not in OFFICIAL_PHYSFORMER}
    assert len(defaults) == 30 + 2 + 5  # + mmpd_to_kr train/eval and the 5 preprocess configs
    assert all(value is None for value in defaults.values()), {rel: v for rel, v in defaults.items() if v is not None}


@pytest.mark.parametrize("rel, expected", [
    ("pure_to_kr/physformer.yaml", 2.0),
    ("ubfc_rppg_to_kr/physformer.yaml", 2.0),
    ("kr_to_ubfc_phys/physformer_fold0.yaml", 0.7),
    ("kr_to_pure/physformer_fold4.yaml", 0.7),
    ("kr_kfold/train/fold0_physformer.yaml", 0.7),
    ("kr_kfold/eval/fold2_physformer.yaml", 0.7),
])
def test_toolbox_resolves_the_physformer_gra_sharp(rel, expected, tmp_path, monkeypatch):
    toolbox = REPO / "third_party" / "rPPG-Toolbox"
    sys.path.insert(0, str(toolbox))
    from config import get_config

    trainer = (toolbox / "neural_methods" / "trainer" / "PhysFormerTrainer.py").read_text()
    rule = "self.gra_sharp = gra_cfg if gra_cfg >= 0 else float(config.MODEL.PHYSFORMER.THETA)"
    assert rule in trainer  # the rule applied below
    for name in PLACEHOLDERS:
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    cfg = get_config(Namespace(config_file=str(CONFIGS / rel)))
    gra_cfg = float(cfg.MODEL.PHYSFORMER.GRA_SHARP)
    assert (gra_cfg < 0) == (expected != 2.0)  # KR-rPPG models: default, i.e. THETA
    assert (gra_cfg if gra_cfg >= 0 else float(cfg.MODEL.PHYSFORMER.THETA)) == pytest.approx(expected)


def test_config_patterns_expand_from_any_folder():
    import common

    pure = common.expand_configs(["preprocess/pure_*.yaml"])  # quoted pattern, relative to benchmark/configs/
    assert [p.name for p in pure] == [f"pure_{m}.yaml" for m in sorted(common.MODELS)]
    absolute = common.expand_configs([str(CONFIGS / "preprocess" / "mmpd_*.yaml")])
    assert len(absolute) == 5 and all(p.parent == CONFIGS / "preprocess" for p in absolute)
    with pytest.raises(SystemExit):
        common.expand_configs(["preprocess/nothing_*.yaml"])


def test_preprocess_configs_match_supported_loaders():
    import preprocess

    datasets = set()
    for path in sorted((CONFIGS / "preprocess").glob("*.yaml")):
        data = yaml.safe_load(path.read_text())["TEST"]["DATA"]
        assert data["DO_PREPROCESS"] is True, path.name
        assert data["DATASET"] in preprocess.LOADERS, path.name
        datasets.add(data["DATASET"])
        module, _ = preprocess.LOADERS[data["DATASET"]]
        assert (REPO / "third_party" / "rPPG-Toolbox" / (module.replace(".", "/") + ".py")).is_file()
    assert datasets == set(preprocess.LOADERS)
