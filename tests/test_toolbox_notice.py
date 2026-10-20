"""The rPPG-Toolbox adaptation documents every change (NOTICE.md, CHANGES/), ships no file of GPL-3.0 ancestry,
and loads checkpoints safely."""

import ast
import re
import sys
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
TOOLBOX = REPO / "third_party" / "rPPG-Toolbox"
NOTICE = (TOOLBOX / "NOTICE.md").read_text()
DOCS = {"LICENSE", "NOTICE.md", "README.md", "USE_RESTRICTIONS.md"}
NEW_FILES = {"dataset/data_loader/KoreanrPPGLoader.py", "neural_methods/loss/PhysFormerLossAdapter.py"}
sys.path.insert(0, str(TOOLBOX))

from neural_methods.loss.PhysFormerLossAdapter import UPSTREAM_RELPATH  # noqa: E402

# Upstream files whose header says "modified based on" hr-cnn (GPL-3.0): never part of this repository. The
# PhysFormer loss may sit in a working copy after tools/fetch_physformer_loss.py; it is not counted here.
NOT_DISTRIBUTED = {UPSTREAM_RELPATH, "neural_methods/loss/RythmFormerLossComputer.py"}
FETCHED = {UPSTREAM_RELPATH, UPSTREAM_RELPATH + ".part"}


def _toolbox_files():
    return sorted(p.relative_to(TOOLBOX).as_posix() for p in TOOLBOX.rglob("*")
                  if p.is_file() and p.relative_to(TOOLBOX).parts[0] != "CHANGES" and "__pycache__" not in p.parts
                  and p.relative_to(TOOLBOX).as_posix() not in FETCHED)


def _patched_files():
    out = []
    for patch in sorted((TOOLBOX / "CHANGES").glob("*.patch")):
        targets = re.findall(r"^\+\+\+ b/(\S+)", patch.read_text(), flags=re.MULTILINE)
        assert len(targets) == 1, patch.name
        out.append(targets[0])
    return out


def test_every_patched_file_exists_and_is_named_in_notice():
    for rel in _patched_files():
        assert (TOOLBOX / rel).is_file(), rel
        assert rel in NOTICE or f"`{Path(rel).name}`" in NOTICE, rel


def test_notice_counts_match_the_folder():
    files = _toolbox_files()
    patched = _patched_files()
    unsupervised = [f for f in files if f.startswith("unsupervised_methods/")]
    match = re.search(r"the (\d+) files of `unsupervised_methods/`", NOTICE)
    assert match and int(match.group(1)) == len(unsupervised)
    match = re.search(r"holds (\d+) files besides `CHANGES/`: (\d+) unmodified upstream files, (\d+) modified files"
                      r".*?(\d+) new files", NOTICE, flags=re.DOTALL)
    assert match, "summary line missing in NOTICE.md"
    total, unmodified, modified, new = map(int, match.groups())
    assert total == len(files)
    assert modified == len(patched)
    assert new == len(NEW_FILES) and NEW_FILES <= set(files)
    assert all(f"`{name}`" in NOTICE for name in NEW_FILES)
    assert unmodified == len(files) - len(patched) - len(DOCS) - len(NEW_FILES)


def test_no_file_of_gpl_ancestry_is_shipped():
    files = _toolbox_files()
    assert not NOT_DISTRIBUTED & set(files)
    assert not any(Path(rel).name in {Path(n).name for n in NOT_DISTRIBUTED} for rel in _patched_files())
    for rel in files:
        if rel.endswith(".py"):
            text = (TOOLBOX / rel).read_text(errors="replace")
            assert "radimspetlik/hr-cnn" not in text, rel  # the upstream header of both files names this URL
    for name in NOT_DISTRIBUTED:
        assert f"`{name}`" in NOTICE, name  # documented as not included


@pytest.mark.parametrize("folder", ["third_party/rPPG-Toolbox", "benchmark", "checkpoints", "analysis"])
def test_every_torch_load_is_weights_only(folder):
    calls = 0
    for path in sorted((REPO / folder).rglob("*.py")):
        with warnings.catch_warnings():  # some upstream files have invalid escape sequences
            warnings.simplefilter("ignore", (DeprecationWarning, SyntaxWarning))
            tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"):
                calls += 1
                flags = {kw.arg: kw.value for kw in node.keywords}
                value = flags.get("weights_only")
                assert isinstance(value, ast.Constant) and value.value is True, f"{path}:{node.lineno}"
    if folder == "third_party/rPPG-Toolbox":
        assert calls == 19  # the five trainers
