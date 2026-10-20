# NOTICE: KR-rPPG adaptation of rPPG-Toolbox

This folder is a **derivative work of rPPG-Toolbox** and stays under the upstream licence,
the **Responsible Artificial Intelligence Source Code License, Version 1.1 (Nov 20, 2022)**.
The complete licence text is in [`LICENSE`](LICENSE) (byte-identical to the upstream file,
sha256 `a2d16c386a18fc6c88ce0ff9d4593082f74a041c96e92ca9e2d89731955f1e1c`). The use restrictions of
Section 3.2 apply to this folder and to anything derived from it; see
[`USE_RESTRICTIONS.md`](USE_RESTRICTIONS.md). The MIT licence at the root of this repository does
**not** apply to this folder.

## Upstream

| | |
|---|---|
| Project | rPPG-Toolbox, https://github.com/ubicomplab/rPPG-Toolbox |
| Base commit | `b7500b848f84ad7f86e277b4612563b69f4f88f9` (2025-09-15, "Support LADH and SUMS Datasets (#405)") |
| Licence | Responsible AI Source Code License v1.1 (upstream `LICENSE`, unchanged since 2023-06-06) |
| Copyright | The rPPG-Toolbox contributors (the licence file carries no copyright line) |
| Citation requested by upstream | Liu, X. et al. rPPG-Toolbox: Deep Remote PPG Toolbox. NeurIPS 2023 (Datasets and Benchmarks). |

Only the files the KR-rPPG benchmark needs are included. Not included: other models, loaders and
configs, `final_model_release/` (official weights: download them from upstream, see
`checkpoints/checkpoints_manifest.csv`), the YOLO5Face face detector
(`dataset/data_loader/face_detector/`, ported from deepcam-cn/yolov5-face, GPL-3.0), and the two loss files
whose header states that they were "modified based on" https://github.com/radimspetlik/hr-cnn (GPL-3.0):
`neural_methods/loss/PhysFormerLossComputer.py` and `neural_methods/loss/RythmFormerLossComputer.py`. Every
KR-rPPG config uses the Haar-cascade backend (`BACKEND: HC`); `BaseLoader` imports YOLO5Face only when a
`Y5F` backend is requested and otherwise fails with a clear message. RhythmFormer is not used by KR-rPPG.
PhysFormer needs the upstream PhysFormer loss: `tools/fetch_physformer_loss.py` (repository root) downloads the
unmodified file from this base commit, checks its sha256 and places it in `neural_methods/loss/` (git-ignored,
never committed here) or, with `--dest`, in a file outside the repository that the environment variable
`KR_RPPG_PHYSFORMER_LOSS` names; `PhysFormerTrainer` stops with these instructions while it is missing or
differs from the pinned file.

## Statement of changes

Who: the KR-rPPG authors (2026). What: the files below. Exact line-level changes against the base commit
are in [`CHANGES/`](CHANGES) (one unified diff per modified file, `a/` = upstream, `b/` = this folder).

This folder holds 47 files besides `CHANGES/`: 24 unmodified upstream files, 17 modified files (one patch
each in `CHANGES/`, 17 patches), 2 new files, and 4 documentation files (`LICENSE`, `NOTICE.md`, `README.md`,
`USE_RESTRICTIONS.md`). The upstream PhysFormer loss file, once fetched, is not part of this folder's contents.

### Unmodified upstream files (byte-identical to the base commit; 24)

`dataset/__init__.py`, `dataset/haarcascade_frontalface_default.xml` (OpenCV, Intel licence header kept),
`evaluation/BlandAltmanPy.py`, `evaluation/bigsmall_multitask_metrics.py`, `evaluation/metrics.py`,
`neural_methods/__init__.py`, `neural_methods/loss/{__init__,NegPearsonLoss}.py`,
`neural_methods/model/{__init__,DeepPhys,EfficientPhys,PhysNet,TS_CAN}.py`,
`neural_methods/trainer/BaseTrainer.py`, and the 10 files of `unsupervised_methods/`
(`__init__.py`, `unsupervised_predictor.py`, `utils.py`,
`methods/{CHROME_DEHAAN,GREEN,ICA_POH,LGI,OMIT,PBV,POS_WANG}.py`).

### New files (2)

| File | Purpose |
|---|---|
| `neural_methods/loss/PhysFormerLossAdapter.py` | KR-rPPG wrapper around the upstream PhysFormer loss, which is not distributed here (see above). Checks the sha256 of the fetched `PhysFormerLossComputer.py` (base commit; `neural_methods/loss/` or `$KR_RPPG_PHYSFORMER_LOSS`) and runs exactly the bytes it checked, limits the target heart rate to 40-179 bpm and then calls the upstream `cross_entropy_power_spectrum_DLDL_softmax2`; contains no code of the upstream file. Result-relevant: this is the KR-rPPG change to the PhysFormer loss that was active in the manuscript runs (previously an in-place edit of the upstream file); outputs and gradients are identical to that edit (`tests/test_physformer_loss.py`). |
| `dataset/data_loader/KoreanrPPGLoader.py` | KR-rPPG loader for the BFDT release layout (`pNNN/T0x/video_bfdt_ffv1.avi`, `BVP_synchronized.csv`, `face_bboxes.csv`; the video file name can be changed with `KR_RPPG_VIDEO_FILE`); session-median face box enlarged by `LARGE_BOX_COEF`, first 6 s skipped; k-fold selection from `metadata/fold_assignment.csv` (`KFOLD_CSV`, `KFOLD_ID`, `KFOLD_TEST_FOLD`) and an optional per-fold train/valid list (`KFOLD_SPLIT_CSV`). Adapted from upstream loaders. |

### Modified files (17)

**Result-relevant changes** (they affect trained weights or reported numbers; all were active in the
manuscript runs unless noted):

| File | Change |
|---|---|
| `neural_methods/trainer/PhysFormerTrainer.py` | `gra_sharp` (attention sharpness) in train/valid/test is `MODEL.PHYSFORMER.THETA` (0.7) instead of upstream's 2.0, now selectable with the new `MODEL.PHYSFORMER.GRA_SHARP` (< 0 = use THETA, the value of KR-rPPG k-fold training and of scoring the models trained with this code; 2.0 = upstream's value, used to score the official PURE / UBFC-rPPG weights). Every PhysFormer model is scored with the `gra_sharp` it was trained with: the KR-rPPG fold models and the MMPD models with the default (0.7), the official PURE / UBFC-rPPG checkpoints (trained by upstream at 2.0) with `GRA_SHARP: 2.0`, set in the 2 configs `pure_to_kr/physformer.yaml` and `ubfc_rppg_to_kr/physformer.yaml`. Learning-rate schedule `CosineAnnealingLR(T_max=EPOCHS, eta_min=1e-6)` instead of `StepLR(50, 0.5)`; validation uses a per-sample std; `get_hr` band 40-180 bpm with clipping and fallbacks. The DLDL frequency loss is taken from `PhysFormerLossAdapter.py` (upstream loss with the target heart rate limited to 40-179 bpm, see *New files*), and the trainer stops at construction with fetch instructions if the upstream loss file is missing. |
| `neural_methods/trainer/DeepPhysTrainer.py` | `ConstantLR` instead of `OneCycleLR`; optional `TRAIN.PRETRAINED_MODEL` loading (unused: every model was trained from random initialisation). |
| `neural_methods/trainer/PhysnetTrainer.py` | Skips batches whose prediction or label is (near) constant, skips NaN/Inf losses, clips gradients at norm 5.0; validation guards against zero std and empty loss lists; optional `TRAIN.PRETRAINED_MODEL`. |
| `neural_methods/loss/PhysNetNegPearsonLoss.py` | Denominator clamped (>= 1e-8) and Pearson r clamped to [-1, 1]. |
| `neural_methods/model/PhysFormer.py` | Default `theta` of `CDC_T` 0.6 -> 0.7 (the configs pass THETA explicitly, so no effect in the benchmark). |
| `evaluation/post_process.py` | Smoothness-priors detrend solved with a sparse solver instead of a dense inverse (numerically equivalent, differences ~1e-11); FFT-based MACC. |

**Other changes** (engineering only):

| File | Change |
|---|---|
| `config.py` | New keys `TRAIN.PRETRAINED_MODEL`, `*.DATA.KFOLD_CSV / KFOLD_ID / KFOLD_TEST_FOLD / KFOLD_SPLIT_CSV`, `MODEL.DEEPPHYS.FRAME_DEPTH`, `MODEL.PHYSFORMER.GRA_SHARP` (default -1.0 = use THETA, for KR-rPPG k-fold training and for scoring the models trained with this code; 2.0 = upstream's value, used to score the official PURE / UBFC-rPPG weights; see the `PhysFormerTrainer.py` row); `${VAR}` environment-variable expansion in YAML files, with an error when a variable is unset. |
| `main.py` | `KoreanrPPG` dataset dispatch; seed from `RPPG_SEED` (default 100, as upstream); data-loader workers/prefetch from `RPPG_DATALOADER_WORKERS` / `RPPG_PREFETCH_FACTOR` (upstream: 16 workers), `pin_memory`, persistent workers. |
| `dataset/data_loader/__init__.py`, `neural_methods/trainer/__init__.py` | Import only the loaders/trainers used here. |
| `dataset/data_loader/BaseLoader.py` | `DiffOnly` label type; YOLO5Face imported only for `Y5F` backends; `cv2.setNumThreads(1)`; preprocessing worker count from `PREPROCESS_WORKERS`; a failing preprocessing subprocess raises instead of leaving a partial cache; one example path in a comment generalised. |
| `dataset/data_loader/UBFCPHYSLoader.py` | Streaming crop/resize path that avoids loading full 1024x1024 videos into memory when face detection is fixed to the first frame (output-equivalent; disable with `RPPG_UBFCPHYS_STREAM_PREPROCESS=0`); `USE_PSUEDO_PPG_LABEL` read defensively. |
| `dataset/data_loader/MMPDLoader.py` | Subject folders sorted; `build_file_list_retroactive` matches MMPD cache names (`subject<N>_..._input<k>.npy`). |
| `dataset/data_loader/PURELoader.py`, `UBFCrPPGLoader.py` | Directory listings sorted; PNG paths resolved with `realpath`; comment/typo clean-up. |
| `neural_methods/trainer/EfficientPhysTrainer.py`, `neural_methods/trainer/TscanTrainer.py` | Optional `TRAIN.PRETRAINED_MODEL` loading (unused: every model was trained from random initialisation); `weights_only=True` (next row). |
| all five trainers (`DeepPhysTrainer.py`, `EfficientPhysTrainer.py`, `PhysFormerTrainer.py`, `PhysnetTrainer.py`, `TscanTrainer.py`) | Every `torch.load` call passes `weights_only=True`, so a checkpoint is loaded as tensors only and cannot run code (added in this repository; the manuscript runs used upstream's plain `torch.load`). The benchmark checkpoints are plain `state_dict`s, so they load and score exactly as before. |

## Third-party code inside this folder

* `config.py` starts with the Swin Transformer header ("Copyright (c) 2021 Microsoft, Licensed under The MIT
  License"), inherited from upstream. Its "see LICENSE" refers to the MIT licence, not to the RAIL `LICENSE` of
  this folder; the MIT licence text is in
  [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md#mit-licence-text) at the repository root.
* `neural_methods/model/PhysNet.py` carries its original header (MIT, (c) 2019 Zitong Yu, with the note
  "Only for research purpose, and commercial use is not allowed"); kept verbatim. MIT licence text: as above.
* `neural_methods/model/PhysFormer.py` is based on https://github.com/ZitongYu/PhysFormer (MIT, Copyright (c) 2021
  Zitong Yu; the file has no licence header). MIT licence text: as above.
* `neural_methods/loss/PhysNetNegPearsonLoss.py`: the `Neg_Pearson` loss is from the original PhysNet authors
  (https://github.com/ZitongYu/PhysNet, `NegPearsonLoss.py`), as credited in its docstring (header kept).
* `neural_methods/trainer/PhysFormerTrainer.py` is based on the PhysFormer training script
  (https://github.com/ZitongYu/PhysFormer, `train_Physformer_160_VIPL.py`) and on the PhysBench notebook
  (https://github.com/KegangWangCCNU/PhysBench, `benchmark_addition/PhysFormer_pure.ipynb`), as credited in its
  module docstring (header kept).
* Not in this folder: `neural_methods/loss/PhysFormerLossComputer.py` and
  `neural_methods/loss/RythmFormerLossComputer.py` state that they were "modified based on"
  https://github.com/radimspetlik/hr-cnn (GPL-3.0). They are therefore not redistributed here. The PhysFormer
  loss is obtained from upstream with `tools/fetch_physformer_loss.py` and keeps its upstream terms;
  `neural_methods/loss/PhysFormerLossAdapter.py` uses it without copying any of its code.
* `dataset/haarcascade_frontalface_default.xml`: OpenCV data file under the Intel License Agreement for
  Open Source Computer Vision Library (header kept).
