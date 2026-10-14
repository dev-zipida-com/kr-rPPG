# Third-party notices

| Component | Where | Origin | Licence / terms | Handling here |
|---|---|---|---|---|
| rPPG-Toolbox | `third_party/rPPG-Toolbox/` | https://github.com/ubicomplab/rPPG-Toolbox, commit `b7500b848f84ad7f86e277b4612563b69f4f88f9` | Responsible AI Source Code License v1.1 | Licence text verbatim, changes in `NOTICE.md` / `CHANGES/`, Section 3.2 notice in `USE_RESTRICTIONS.md`. Please cite Liu et al., NeurIPS 2023. |
| Swin Transformer config header | `third_party/rPPG-Toolbox/config.py` | Microsoft (2021), via rPPG-Toolbox | MIT, Copyright (c) 2021 Microsoft (header kept) | Unchanged header; MIT licence text [below](#mit-licence-text). |
| PhysNet model | `third_party/rPPG-Toolbox/neural_methods/model/PhysNet.py` | Z. Yu et al., via rPPG-Toolbox | File header: MIT (c) 2019 Zitong Yu, "Only for research purpose, and commercial use is not allowed" | Header kept verbatim; MIT licence text [below](#mit-licence-text). |
| PhysNet negative Pearson loss | `third_party/rPPG-Toolbox/neural_methods/loss/PhysNetNegPearsonLoss.py` | `NegPearsonLoss.py` of https://github.com/ZitongYu/PhysNet (Z. Yu), via rPPG-Toolbox | As distributed by rPPG-Toolbox (RAIL v1.1); the origin is credited in the file | Header kept (source URL in the class docstring); KR-rPPG changes in `CHANGES/`. |
| PhysFormer trainer | `third_party/rPPG-Toolbox/neural_methods/trainer/PhysFormerTrainer.py` | Based on `train_Physformer_160_VIPL.py` of https://github.com/ZitongYu/PhysFormer and the PhysBench notebook `benchmark_addition/PhysFormer_pure.ipynb` of https://github.com/KegangWangCCNU/PhysBench, via rPPG-Toolbox | As distributed by rPPG-Toolbox (RAIL v1.1); both origins are credited in the file | Header kept (module docstring); KR-rPPG changes in `CHANGES/`. |
| PhysFormer model | `third_party/rPPG-Toolbox/neural_methods/model/PhysFormer.py` | `Physformer.py` and `transformer_layer.py` of https://github.com/ZitongYu/PhysFormer (as stated in the file's docstring), via rPPG-Toolbox | MIT, Copyright (c) 2021 Zitong Yu (`LICENSE` of the PhysFormer repository); the file has no licence header | Docstring kept; MIT licence text [below](#mit-licence-text). |
| PhysFormer loss (`neural_methods/loss/PhysFormerLossComputer.py` of rPPG-Toolbox) | **not included**; the user fetches it with `tools/fetch_physformer_loss.py` (rPPG-Toolbox commit `b7500b8`, sha256-checked) into `third_party/rPPG-Toolbox/neural_methods/loss/` (git-ignored) or into a file outside the repository named by `KR_RPPG_PHYSFORMER_LOSS` | rPPG-Toolbox; its header states "modified based on" https://github.com/radimspetlik/hr-cnn | hr-cnn is GPL-3.0; rPPG-Toolbox distributes the file under RAIL v1.1 | Not redistributed because of the stated GPL-3.0 ancestry. The KR-rPPG change (target heart rate limited to 40-179 bpm) is applied by the new `third_party/rPPG-Toolbox/neural_methods/loss/PhysFormerLossAdapter.py`, which calls the upstream function and contains none of its code. |
| RhythmFormer loss (`neural_methods/loss/RythmFormerLossComputer.py` of rPPG-Toolbox) | not included | rPPG-Toolbox; same "modified based on" hr-cnn statement | hr-cnn is GPL-3.0 | Not used by KR-rPPG (no RhythmFormer runs); not redistributed. |
| OpenCV Haar cascade | `third_party/rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml` | OpenCV | Intel License Agreement for Open Source Computer Vision Library (BSD-style) | Header kept. |
| YOLO5Face face detector | not included | deepcam-cn/yolov5-face via rPPG-Toolbox | GPL-3.0 | Excluded; only needed for the `Y5F` backend, which KR-rPPG does not use. |
| Official rPPG-Toolbox weights (PURE, UBFC-rPPG) | not included | rPPG-Toolbox `final_model_release/` at `b7500b8` | RAIL v1.1; trained on research-only datasets | Linked with sha256 in `checkpoints/checkpoints_manifest.csv`, not re-hosted. |
| BFDT de-identification | `release_builder/krrppg_release/bfdt.py`, `privacy/bfdt_transform.py` | Method of Huang, Ou & Xiong, Biomed. Signal Process. Control 108, 107893 (2025), https://doi.org/10.1016/j.bspc.2025.107893 | Re-implemented (MIT); please cite the paper | - |
| InsightFace `buffalo_l` (Table 9 attack) | not included | https://github.com/deepinsight/insightface | Code MIT; pretrained models non-commercial research only | Not redistributed; the Table 9 attack code is not part of this repository (see README). |

## Datasets used by the benchmark (not included, obtain from their owners)

| Dataset | Terms (check the current terms with the owners) |
|---|---|
| KR-rPPG | Data use agreement (DUA); controlled-access repository (link, DOI and access procedure to be added on publication) |
| PURE | Research use; request from TU Ilmenau |
| UBFC-rPPG | Research use; see the dataset site |
| MMPD | Academic use only, signed release agreement; commercial use banned |
| UBFC-Phys | Research use; see the dataset site |

Font files are not redistributed; `analysis/figure3/make_figure3.py` uses matplotlib's default fonts.

## MIT licence text

Three files in `third_party/rPPG-Toolbox/` come from MIT-licensed code and do not contain the licence text:
`config.py` (Swin Transformer header, "Copyright (c) 2021 Microsoft, Licensed under The MIT License"),
`neural_methods/model/PhysNet.py` (header "MIT License, Copyright (c) 2019", by Zitong Yu) and
`neural_methods/model/PhysFormer.py` (no licence header; its docstring names https://github.com/ZitongYu/PhysFormer,
whose `LICENSE` is the MIT licence with "Copyright (c) 2021 Zitong Yu"; `neural_methods/trainer/PhysFormerTrainer.py`
is based on a training script of the same repository). The `LICENSE` file in that folder is the rPPG-Toolbox licence
(RAIL v1.1), not the MIT licence these notices refer to; the MIT licence text for the three notices is:

```
Copyright (c) 2021 Microsoft
Copyright (c) 2019 Zitong Yu
Copyright (c) 2021 Zitong Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The PhysNet header also states "Only for research purpose, and commercial use is not allowed".
