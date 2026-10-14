"""Public-repository checks: no trained KR-rPPG weights are listed or offered (the manifest holds only the
official rPPG-Toolbox weights, linked with sha256), and the reader-facing documents carry no TODO markers and no
instructions to download weights from GitHub."""

import csv
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "checkpoints" / "checkpoints_manifest.csv"
PUBLIC_COLUMNS = ["file", "architecture", "training_data", "bytes", "sha256", "n_params"]
DOCS = ("README.md", "CITATION.cff", "LICENSES.md", "THIRD_PARTY_NOTICES.md", "checkpoints/README.md",
        "privacy/README.md", "release_builder/README.md", "third_party/rPPG-Toolbox/README.md",
        "third_party/rPPG-Toolbox/NOTICE.md", "third_party/rPPG-Toolbox/USE_RESTRICTIONS.md")
# Wording that would list or offer KR-rPPG-trained weights (the public repository offers none).
WEIGHT_OFFERS = (r"\bon request\b", r"available under the dua",
                 r"dua[- ]approved", r"\bdua recipients\b", r"fold weights", r"weights \(dua\)",
                 r"trained weights are available", r"obtained the weights")


def _manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_manifest_has_only_public_columns():
    columns, _ = _manifest()
    assert columns == PUBLIC_COLUMNS


def test_manifest_lists_only_the_official_rppg_toolbox_weights():
    _, rows = _manifest()
    assert len(rows) == 10
    assert all(row["file"].startswith("upstream:final_model_release/") for row in rows)
    assert {row["training_data"] for row in rows} == {"PURE", "UBFC-rPPG"}


def test_manifest_values_are_well_formed():
    _, rows = _manifest()
    for row in rows:
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"]), row["file"]
        assert int(row["bytes"]) > 0 and int(row["n_params"]) > 0, row["file"]
        assert row["file"].endswith(".pth"), row["file"]
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert len({row["file"] for row in rows}) == len(rows)


def test_readme_names_the_repository_url_of_citation_cff():
    cff = (REPO / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r'^repository-code: "(https://github\.com/[\w.-]+/[\w.-]+)"$', cff, flags=re.MULTILINE)
    assert match, "repository-code missing in CITATION.cff"
    assert match.group(1) in (REPO / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", DOCS)
def test_reader_facing_documents_have_no_todo_markers(doc):
    text = (REPO / doc).read_text(encoding="utf-8")
    assert not re.search(r"\bTODO\b", text)


@pytest.mark.parametrize("doc", DOCS)
def test_no_instructions_to_download_weights_from_github(doc):
    text = (REPO / doc).read_text(encoding="utf-8").lower()
    assert "gh release" not in text and "releases/download" not in text and "release asset" not in text


@pytest.mark.parametrize("doc", DOCS)
def test_documents_do_not_offer_kr_rppg_trained_weights(doc):
    text = " ".join((REPO / doc).read_text(encoding="utf-8").lower().split())
    found = [pattern for pattern in WEIGHT_OFFERS if re.search(pattern, text)]
    assert not found, found
