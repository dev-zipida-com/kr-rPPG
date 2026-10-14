# Licences in this repository

This repository mixes code under two licences. The licence of a file is determined by its folder:

| Path | Licence | Notes |
|---|---|---|
| everything **outside** `third_party/` (`release_builder/`, `benchmark/`, `analysis/`, `privacy/`, `checkpoints/*.py`, `tools/`, `tests/`, docs, configs) | **MIT** ([`LICENSE`](LICENSE)), Copyright (c) 2026 ZIPIDA and GIST | New code written for KR-rPPG. It may *call* the toolbox code below, but contains no copy of it. |
| `third_party/rPPG-Toolbox/` | **Responsible AI Source Code License v1.1** ([`third_party/rPPG-Toolbox/LICENSE`](third_party/rPPG-Toolbox/LICENSE), verbatim upstream text) | KR-rPPG adaptation of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) at commit `b7500b8`. Changes are listed in [`NOTICE.md`](third_party/rPPG-Toolbox/NOTICE.md) and [`CHANGES/`](third_party/rPPG-Toolbox/CHANGES). The **use restrictions of Section 3.2** (no inference of health/medical conditions or age, no insurance decisions, no diagnosis without human oversight, no surveillance or criminal prediction) apply: see [`USE_RESTRICTIONS.md`](third_party/rPPG-Toolbox/USE_RESTRICTIONS.md). |

Running the benchmark executes the RAIL-licensed toolbox code, so in practice **anyone running the
benchmark is bound by the RAIL Section 3.2 restrictions**, even though the scripts that drive it are MIT.
If you copy code out of `third_party/rPPG-Toolbox/` into another file, that file becomes a derivative
work under RAIL v1.1, not MIT.

## Not covered by either licence

* **The KR-rPPG dataset** (videos, physiological signals, metadata, fold assignment) is not part of this
  repository. It is distributed separately under a data use agreement (DUA); see the README.
* **Model weights** are not part of this repository. Users train the KR-rPPG models themselves with the
  provided configs. The official rPPG-Toolbox weights used for two Table 8 rows are linked, with their sha256
  in [`checkpoints/checkpoints_manifest.csv`](checkpoints/checkpoints_manifest.csv), not re-hosted.
* **rPPG-Toolbox's `neural_methods/loss/PhysFormerLossComputer.py`** is not part of this repository: its header
  states that it was "modified based on" hr-cnn (GPL-3.0). [`tools/fetch_physformer_loss.py`](tools/fetch_physformer_loss.py)
  downloads the unmodified file from upstream into your working copy (git-ignored) or into a file outside it that
  you name; it stays under the terms that apply to it upstream and must not be committed or redistributed with
  this repository. The KR-rPPG change to that loss is applied by
  `third_party/rPPG-Toolbox/neural_methods/loss/PhysFormerLossAdapter.py` (RAIL v1.1 like the rest of that
  folder), which contains no code of the upstream file.
* **Third-party components and datasets** referenced by the code: see
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
