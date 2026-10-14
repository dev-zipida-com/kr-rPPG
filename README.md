# KR-rPPG: code for the dataset release and the benchmark

Code accompanying the data descriptor *"KR-rPPG: a clinical remote photoplethysmography dataset"*
(Choi, J. et al., 2026; DOI to be added on publication). KR-rPPG contains 1,166 synchronized sessions (talking, stationary,
yaw and pitch rotation) from 296 Korean outpatients recorded at three hospital sites, released as
BFDT de-identified videos with contact BVP, heart rate, SpO2, frame timestamps, face boxes and
participant metadata.

This repository contains:

* `release_builder/` - the tool that builds and verifies the de-identified (BFDT) release (MIT);
* `third_party/rPPG-Toolbox/` - the KR-rPPG adaptation of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox)
  used for every benchmark number (**Responsible AI Source Code License v1.1**, not MIT; see below);
* `benchmark/` - configs and scripts for Table 8 (MIT);
* `analysis/` - Table 7 and Figure 3 (MIT);
* `checkpoints/` - sha256 manifest of the official rPPG-Toolbox weights used for two Table 8 rows and a
  verification script; **no trained weights are distributed here** (see [Checkpoints](#checkpoints));
* `privacy/` - reference BFDT implementation (MIT);
* `tools/` - `fetch_physformer_loss.py`, which fetches the one rPPG-Toolbox file that is not distributed here
  (see [PhysFormer loss](#physformer-loss-fetched-separately)) (MIT).

> **Licences in one sentence:** our own code is MIT; everything in `third_party/rPPG-Toolbox/` keeps the
> rPPG-Toolbox licence (RAIL v1.1), whose Section 3.2 forbids, among other things, using the code or models
> derived from it to infer health/medical conditions or age, for insurance decisions, or to diagnose without
> human oversight. Details: [`LICENSES.md`](LICENSES.md).

## What can be reproduced, and by whom

| Manuscript item | Code | Needs | Reproducible by |
|---|---|---|---|
| Release build (BFDT videos, metadata) | `release_builder/` | original recordings + CRF (custodian only) | data custodian; anyone can run `verify` on the released data and the unit tests on synthetic data |
| Table 7 (stationary HR distributions) | `analysis/table7/` | KR-rPPG prediction table of the internal row (DUA) + raw PURE, MMPD, UBFC-Phys | DUA users (the PURE / MMPD / UBFC-Phys rows: anyone with those datasets) |
| Table 8, internal 5-fold row | `benchmark/` (`kr_kfold`) | KR-rPPG (DUA); fold models you train with `kr_kfold/train` | DUA users, on the released BFDT videos (numbers differ, see caveats) |
| Table 8, KR-rPPG -> PURE / UBFC-rPPG / MMPD / UBFC-Phys | `benchmark/` (`kr_to_*`) | the fold models from the internal row + the public dataset | DUA users, with models trained on the BFDT videos (numbers differ, see caveats) |
| Table 8, PURE / UBFC-rPPG -> KR-rPPG | `benchmark/` (`*_to_kr`) | KR-rPPG (DUA) + official rPPG-Toolbox weights | DUA users (on BFDT videos) |
| Table 8, MMPD -> KR-rPPG | `benchmark/` (`mmpd_to_kr`) | MMPD + KR-rPPG (DUA); models retrained on MMPD | DUA users (on BFDT videos) |
| Figure 3 | `analysis/figure3/` | out-of-fold predictions of the internal row | DUA users |
| Table 9 (ArcFace re-identification) | not included | original, non-de-identified videos | data custodian only; the attack code is not part of this repository |

## Repository layout

```
LICENSE                       MIT (own code)
LICENSES.md                   which licence applies where
THIRD_PARTY_NOTICES.md        third-party components and datasets
CITATION.cff
environment/                  pinned requirements (benchmark: Python 3.10; builder: Python 3.9)
release_builder/              krrppg_release package, tests on synthetic data, README
third_party/rPPG-Toolbox/     RAIL v1.1: LICENSE (verbatim), NOTICE.md, USE_RESTRICTIONS.md, CHANGES/*.patch, code
                              (without the PhysFormer loss file, see below)
benchmark/
  make_configs.py             writes benchmark/configs/ (committed; ${PLACEHOLDER} paths only)
  configs/                    195 YAML configs: preprocess/, kr_kfold/{train,eval}/, kr_to_*/, *_to_kr/, mmpd_to_kr/
  preprocess.py run.py        build caches, run configs (train / evaluate)
  collect_predictions.py      toolbox output pickles -> per-session HR table (kept in $KR_RPPG_WORK)
  make_mmpd_split.py          MMPD subjects 1-26 / 27-33 file lists
  summarize_table8.py         prints Table 8 next to the manuscript values
analysis/figure3/             figure3_values.py, make_figure3.py, expected_values.json (aggregates)
analysis/table7/              table7_hr_distributions.py (PURE / MMPD / UBFC-Phys rows from the raw reference waveforms,
                              KR-rPPG row from the step 2 prediction table; statistics; comparison with the manuscript)
checkpoints/                  checkpoints_manifest.csv (official rPPG-Toolbox weights), verify_checkpoints.py, README.md
privacy/                      bfdt_transform.py (reference BFDT), README.md
tools/                        fetch_physformer_loss.py (fetches the upstream PhysFormer loss, sha256-checked)
tests/                        model smoke tests, config checks, synthetic loader and analysis tests, PhysFormer loss
```

## Data access

* **KR-rPPG**: the data are available to approved researchers under a data use agreement (DUA) from a
  controlled-access repository; the link, DOI and access procedure will be added on publication. The release
  folder `KR-rPPG/` contains
  `p001/ ... p296/` with task folders `T00`-`T03` (`video_bfdt_ffv1.avi`, `BVP_synchronized.csv`,
  `HR_synchronized.csv`, `SpO2_synchronized.csv`, `frames_timestamp.csv`, `face_bboxes.csv`) and
  `metadata/` (`participants.csv`, `participants_dictionary.csv`, `task_session_availability.csv`,
  `fold_assignment.csv`). Nothing from the dataset is, or may be, committed to this repository.
* **PURE**, **UBFC-rPPG**, **MMPD**, **UBFC-Phys**: obtain them from their owners under their terms
  (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)). The loaders expect the folder layouts of
  upstream rPPG-Toolbox (PURE `RawData/`, UBFC-rPPG `RawData/` with `subject*`, MMPD with `subject*`
  `.mat` files, UBFC-Phys `RawData/s*/vid_s*_T*.avi`).

## Installation

```bash
conda env create -f environment/environment.yml && conda activate kr-rppg     # benchmark (Python 3.10, CUDA 12.1)
# or: pip install -r environment/requirements-benchmark.txt --extra-index-url https://download.pytorch.org/whl/cu121
python tools/fetch_physformer_loss.py                                           # needed for PhysFormer, see below
python -m pytest tests                                                          # ~2 min on CPU
```

The release builder has its own requirements (`environment/requirements-release-builder.txt`) and tests
(`cd release_builder && python -m pytest`); see [`release_builder/README.md`](release_builder/README.md).

### PhysFormer loss (fetched separately)

The header of rPPG-Toolbox's `neural_methods/loss/PhysFormerLossComputer.py` states that the file was
"modified based on" [hr-cnn](https://github.com/radimspetlik/hr-cnn), which is licensed under GPL-3.0. This
repository therefore does **not** distribute that file. PhysFormer training and evaluation need it:

```bash
python tools/fetch_physformer_loss.py            # download from rPPG-Toolbox b7500b8, check sha256, install
python tools/fetch_physformer_loss.py --check    # is it there and unmodified?
```

The script downloads the unmodified file from rPPG-Toolbox at commit `b7500b848f84ad7f86e277b4612563b69f4f88f9`
(one anonymous HTTPS request to `raw.githubusercontent.com`), checks its sha256 and places it in
`third_party/rPPG-Toolbox/neural_methods/loss/`, where the toolbox imports it. It writes nothing whose sha256
differs from the pinned value and does not replace a differing file without `--force`. Without network access,
copy the file from a clone of rPPG-Toolbox at that commit and run
`python tools/fetch_physformer_loss.py --from-file <path>`. To keep the file outside the repository, install it
with `--dest /absolute/path/PhysFormerLossComputer.py` and set `KR_RPPG_PHYSFORMER_LOSS` to that path; the toolbox
then reads it from there. The file is git-ignored: do not commit or redistribute it with this repository; it
remains under the terms that apply to it upstream (see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)).

The KR-rPPG change to this loss (target heart rate limited to 40-179 bpm before the distribution and
cross-entropy terms) is applied by `third_party/rPPG-Toolbox/neural_methods/loss/PhysFormerLossAdapter.py`, which
calls the upstream function and contains none of its code. At run time the adapter checks the sha256 of the
file and runs exactly the bytes it checked. `tests/test_physformer_loss.py` checks, once the file has been
fetched, that this reproduces the loss of the manuscript runs. Without the file (or with a different one),
PhysFormer training and evaluation stop at start-up with these instructions (`benchmark/run.py` checks before
starting any config); the other four models do not need it.

## Environment variables

All configs use placeholders that the toolbox expands at run time (it stops if one is unset):

| Variable | Meaning |
|---|---|
| `KR_RPPG_ROOT` | the released `KR-rPPG/` folder (holds `pNNN/` and `metadata/`) |
| `KR_RPPG_WORK` | a private work folder for caches, runs, logs and per-session results (keep it on DUA-compliant storage) |
| `KR_RPPG_CHECKPOINTS` | folder for the models you train: `kr_5fold/<model>/` (step 2; used by `kr_kfold/eval` and `kr_to_*`) and `mmpd_trained/<model>/` (step 5) |
| `RPPG_TOOLBOX_WEIGHTS` | `final_model_release/` of rPPG-Toolbox at commit `b7500b848f84ad7f86e277b4612563b69f4f88f9` |
| `PURE_ROOT`, `UBFC_RPPG_ROOT`, `MMPD_ROOT`, `UBFC_PHYS_ROOT` | raw folders of the public datasets |

Optional: `RPPG_SEED` (default 100, as in the manuscript), `RPPG_DATALOADER_WORKERS` (4),
`RPPG_PREFETCH_FACTOR` (2), `PREPROCESS_WORKERS` (8), `KR_RPPG_PHYSFORMER_LOSS` (absolute path of the fetched
PhysFormer loss if it is kept outside the repository, see [PhysFormer loss](#physformer-loss-fetched-separately)),
`KR_RPPG_VIDEO_FILE` (name of the video file in each KR-rPPG session folder, default `video_bfdt_ffv1.avi`).

## Checkpoints

**No trained weights are distributed with this repository.** Train the KR-rPPG models yourself with the
provided configs (steps 2 and 5); the evaluation configs name the files they expect below `$KR_RPPG_CHECKPOINTS`.

The PURE -> KR-rPPG and UBFC-rPPG -> KR-rPPG rows (step 4) score the official rPPG-Toolbox weights: download
`final_model_release/` from [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) at commit
`b7500b848f84ad7f86e277b4612563b69f4f88f9` (third-party files, not re-hosted here) and check them against
[`checkpoints/checkpoints_manifest.csv`](checkpoints/checkpoints_manifest.csv):

```bash
python checkpoints/verify_checkpoints.py --upstream $RPPG_TOOLBOX_WEIGHTS --load
```

Load checkpoints only with `torch.load(..., weights_only=True)`. See [`checkpoints/README.md`](checkpoints/README.md).

## Step by step

All commands run from the repository root. Configs can be listed with `python benchmark/run.py <folder> --dry-run`.

**0. Configs.** `benchmark/configs/` is generated by `python benchmark/make_configs.py` (already committed;
`--check` verifies it). The settings match the manuscript runs (verified key by key against the original
configs); only paths, the data layout and the items listed under *Caveats* differ. Every PhysFormer config
scores the model with the attention sharpness it was trained with (caveat 3): `pure_to_kr` and
`ubfc_rppg_to_kr` set `MODEL.PHYSFORMER.GRA_SHARP: 2.0` for the official weights, and the configs that use
KR-rPPG fold models leave it unset, so the trainer uses THETA = 0.7.

**1. Preprocess the caches** (face crop with the released face boxes for KR-rPPG, Haar cascade for the
public datasets; first 6 s of each KR-rPPG session skipped):

```bash
python benchmark/preprocess.py preprocess/kr_physnet.yaml preprocess/kr_physformer.yaml \
       preprocess/kr_deepphys.yaml preprocess/kr_tscan.yaml preprocess/kr_efficientphys.yaml
python benchmark/preprocess.py benchmark/configs/preprocess/pure_*.yaml benchmark/configs/preprocess/ubfc_rppg_*.yaml \
       benchmark/configs/preprocess/mmpd_*.yaml benchmark/configs/preprocess/ubfc_phys_*.yaml   # only the rows you need
```

Names such as `preprocess/kr_physnet.yaml` are resolved below `benchmark/configs/`. Shell wildcards are expanded
by your shell relative to the current folder, so give them the full `benchmark/configs/...` path (as above) or
quote them (`'preprocess/pure_*.yaml'`) and let the script expand them.

**2. Table 8, internal 5-fold row and Figure 3**

```bash
python benchmark/run.py kr_kfold/train --gpu 0      # train the 25 fold models (10 epochs each)
python benchmark/run.py kr_kfold/eval --gpu 0       # out-of-fold inference with $KR_RPPG_CHECKPOINTS/kr_5fold
python benchmark/collect_predictions.py kr_kfold_eval
python benchmark/summarize_table8.py
python analysis/figure3/figure3_values.py --compare analysis/figure3/expected_values.json
python analysis/figure3/make_figure3.py --values $KR_RPPG_WORK/results/figure3_values.json --out figure3.pdf
```

Training saves `KFold_<model>_fold<k>_Epoch<e>.pth` under `$KR_RPPG_WORK/runs/kr_kfold/...`. Copy (or link)
the epochs you want to evaluate to `$KR_RPPG_CHECKPOINTS/kr_5fold/<model>/` under the names used in
`kr_kfold/eval/*.yaml` and `kr_to_*/*.yaml` (step 3), or edit `INFERENCE.MODEL_PATH`. The manuscript scored
PhysNet at the best-validation epoch (9/8/7/8/9), DeepPhys at the last saved epoch (8/7/7/8/9; the training
runs of folds 0-3 ended before epoch 9) and the other models at epoch 9.

**3. Table 8, KR-rPPG -> external datasets** (the fold models from step 2 + public data)

```bash
python benchmark/run.py kr_to_pure kr_to_ubfc_rppg kr_to_mmpd kr_to_ubfc_phys --gpu 0
python benchmark/collect_predictions.py kr_to_pure kr_to_ubfc_rppg kr_to_mmpd kr_to_ubfc_phys
python benchmark/summarize_table8.py               # row value = mean of the 5 fold MAEs
```

**4. Table 8, PURE / UBFC-rPPG -> KR-rPPG** (official rPPG-Toolbox weights)

```bash
python benchmark/run.py pure_to_kr ubfc_rppg_to_kr --gpu 0
python benchmark/collect_predictions.py pure_to_kr ubfc_rppg_to_kr
```

**5. Table 8, MMPD -> KR-rPPG**

```bash
python benchmark/make_mmpd_split.py                # MMPD subjects 1-26 train, 27-33 validation
python benchmark/run.py mmpd_to_kr/train --gpu 0   # train on MMPD (MMPD-trained weights are not distributed)
python benchmark/run.py mmpd_to_kr/eval --gpu 0    # score the trained models on KR-rPPG
python benchmark/collect_predictions.py mmpd_to_kr_eval
python benchmark/summarize_table8.py
```

Training saves `Table8_MMPD_to_KR_<model>_s01_26_Epoch<e>.pth` under `$KR_RPPG_WORK/runs/mmpd_to_kr/...`.
Copy (or link) the best-validation epoch to `$KR_RPPG_CHECKPOINTS/mmpd_trained/<model>/` under the name used
in `mmpd_to_kr/eval/*.yaml`, or edit `INFERENCE.MODEL_PATH`.

**6. Table 7** (needs the step 2 prediction table and the raw public datasets; no GPU)

```bash
python analysis/table7/table7_hr_distributions.py --kr $KR_RPPG_WORK/results/kr_kfold_eval.csv \
    --pure $PURE_ROOT --mmpd $MMPD_ROOT --ubfc-phys $UBFC_PHYS_ROOT
```

The KR-rPPG row is the reference HR (`label_hr`) of the DeepPhys T01 sessions in the step 2 table. The PURE
(steady task), MMPD (`Stationary` recordings, mean per subject) and UBFC-Phys (resting task T1) rows are computed
from the raw contact reference waveforms of the datasets: one HR per recording from the band-passed
(0.75-2.5 Hz) whole recording, see the script's docstring. The script prints the four rows and the KR-rPPG versus
dataset tests (Welch, Mann-Whitney U, Holm) next to the manuscript values; it prints aggregates only and writes
nothing.

**7. Release build and verification** - see [`release_builder/README.md`](release_builder/README.md).
DUA users can check a downloaded release with
`python -m krrppg_release verify --release-root $KR_RPPG_ROOT` (run inside `release_builder/`).

## Expected results (manuscript)

The manuscript numbers were obtained on the **original, non-de-identified videos** (custodian side) with
the documented toolbox changes. Session-level MAE in bpm:

| Training -> test | PhysNet | PhysFormer | DeepPhys | TS-CAN | EfficientPhys |
|---|---|---|---|---|---|
| KR-rPPG -> KR-rPPG (5-fold, internal) | 1.47 | 1.51 | 11.83 | 7.48 | 4.81 |
| KR-rPPG -> PURE | 3.65 | 4.34 | 2.74 | 0.31 | 2.80 |
| KR-rPPG -> UBFC-rPPG | 1.64 | 1.70 | 1.85 | 2.00 | 2.78 |
| KR-rPPG -> MMPD | 8.97 | 8.38 | 21.45 | 15.53 | 15.54 |
| KR-rPPG -> UBFC-Phys | 5.11 | 5.66 | 8.50 | 5.79 | 6.36 |
| PURE -> KR-rPPG | 10.83 | 10.57 | 16.98 | 11.51 | 10.87 |
| UBFC-rPPG -> KR-rPPG | 5.19 | 7.41 | 15.96 | 11.65 | 11.56 |

No reference values are listed here for MMPD -> KR-rPPG; step 5 runs that direction and
`summarize_table8.py` prints it without a comparison value.

Figure 3 (participant-mean MAE, bootstrap 95% CI) per model and task is in
[`analysis/figure3/expected_values.json`](analysis/figure3/expected_values.json) (aggregates only); its
`All` group equals the internal row. Table 7: KR-rPPG T01 reference HR 81.0 +/- 13.8 bpm (N = 293),
MMPD 76.8 +/- 11.7 (N = 33), PURE 65.2 +/- 23.5 (N = 10), UBFC-Phys 81.7 +/- 13.1 (N = 56); KR-rPPG versus each
dataset: Welch p >= 0.060 for all three, Mann-Whitney p = 0.001 for PURE (Holm-adjusted 0.003) and not
significant after Holm for MMPD and UBFC-Phys.
Table 9 (privacy evaluation) cannot be reproduced with this repository: the evaluation is described in the
paper and needs the original, non-de-identified videos. The protection of the released data relies on the
BFDT transform together with controlled access under the DUA.

### Caveats: why your numbers can differ

1. **Video source.** Released videos are BFDT de-identified (8 x 8 blocks, keep ratio 0.02); the
   manuscript KR-rPPG runs used the original videos. Rows that train or test on KR-rPPG, including the
   KR-rPPG -> external rows (their models are trained on KR-rPPG), will therefore differ when run on the
   release.
2. **Train/validation split.** In each fold the manuscript used the last 10% of the training pool sorted
   by the original participant keys as validation set. Release IDs are a random permutation, so the default
   split here (last 10% of the pool sorted by release ID) selects different validation participants. An
   explicit per-fold train/validation list keyed by release IDs (columns `fold`, `participant_key`,
   `split`) can be passed with `KFOLD_SPLIT_CSV`. Test folds are identical (`metadata/fold_assignment.csv`).
3. **PhysFormer attention sharpness.** The k-fold trainer trains PhysFormer with `gra_sharp = THETA = 0.7`
   (upstream: 2.0). Every PhysFormer model is scored with the `gra_sharp` it was trained with: 2.0 for the
   official PURE / UBFC-rPPG weights (PURE -> KR, UBFC-rPPG -> KR) and 0.7 for the KR-rPPG fold models (internal
   and KR -> external rows). The configs set this with `MODEL.PHYSFORMER.GRA_SHARP` (2.0 in
   `pure_to_kr/physformer.yaml` and `ubfc_rppg_to_kr/physformer.yaml`; unset elsewhere, which means THETA). Scoring
   with another value changes the PhysFormer results (see `third_party/rPPG-Toolbox/NOTICE.md`).
4. **Epochs per cell.** The internal row and the KR -> external rows use different PhysNet epochs in
   folds 1-3 (best validation vs. epoch 9); the configs name the exact files.
5. **KR-rPPG test set size.** PURE -> KR and UBFC-rPPG -> KR are reported on the 1,166 released sessions;
   12 recorded sessions are not released. On all 1,178 sessions the same predictions give
   10.84 / 10.65 / 16.91 / 11.50 / 10.84 (PURE -> KR) and 5.22 / 7.41 / 15.92 / 11.64 / 11.56 (UBFC-rPPG -> KR).
6. **External caches.** Some external caches used for the manuscript were built with separate preprocessing
   scripts; `benchmark/preprocess.py` uses the toolbox loaders, so the cached sessions can differ. The PURE
   PhysFormer cache (KR-rPPG -> PURE) covered 37 of 59 sessions; the KR-rPPG -> UBFC-rPPG test caches held all
   42 subjects but fewer chunks per session than the full videos give (PhysNet 269, PhysFormer 319 chunks in
   total); the KR-rPPG -> MMPD PhysNet value 8.97 was obtained with a cache that was later rebuilt (8.75 on the
   rebuilt cache). KR-rPPG -> PURE EfficientPhys is 2.805 before rounding.
7. **Training exclusions.** The original k-fold configs listed two sessions under `EXCLUSION_LIST`, but the
   KR-rPPG loader never applied that list, so it is omitted here without changing the experiment.
8. **Randomness.** `RPPG_SEED=100` and deterministic cuDNN are set, but GPU kernels and library versions
   can still change trained weights. Bootstrap CIs (Figure 3) depend on participant order and can move in
   the second decimal with release IDs.
9. **Table 7.** Table 7 is reproduced by `analysis/table7/` from the raw reference waveforms of PURE, MMPD and
   UBFC-Phys (dataset folders supplied by the user) and the `label_hr` column of the step 2 prediction table:
   `python analysis/table7/table7_hr_distributions.py --kr $KR_RPPG_WORK/results/kr_kfold_eval.csv --pure $PURE_ROOT
   --mmpd $MMPD_ROOT --ubfc-phys $UBFC_PHYS_ROOT` (step 6). With the predictions of the manuscript's internal row it
   reproduces all 32 cells of Table 7 and the reported test results. The PURE / MMPD / UBFC-Phys reference HRs of
   Table 7 are not the `label_hr` values of the `kr_to_*` tables: Table 7 takes one HR per raw recording
   (order-1 Butterworth 0.75-2.5 Hz, periodogram of the whole recording) and all 56 UBFC-Phys resting recordings,
   whereas the toolbox post-processing of Table 8 works on the preprocessed labels (resampled to the video frames,
   cut into whole chunks), detrends them and filters at 0.6-3.3 Hz, and the toolbox UBFC-Phys runs exclude 14
   resting recordings.
10. **UBFC-Phys -> KR-rPPG.** Configs for this direction are not part of this repository.

## Tests

```bash
python -m pytest tests                                   # models (random weights), configs, KR loader, analysis, PhysFormer loss
(cd release_builder && python -m pytest)                 # builder (synthetic data, Python 3.9 env)
```

The tests use synthetic data and random weights only. The PhysFormer-loss reproduction tests are skipped until
`python tools/fetch_physformer_loss.py` has been run; without CUDA they run on the CPU.

## Citation

```
Data descriptor: Choi, J. et al. KR-rPPG: a clinical remote photoplethysmography dataset. (2026). DOI to be added
                 on publication.
Dataset:         Choi, J. et al. KR-rPPG: a clinical remote photoplethysmography dataset. (2026). Controlled access
                 under a data use agreement (DUA); repository and DOI to be added on publication.
```

Please also cite rPPG-Toolbox (Liu, X. et al., NeurIPS 2023, https://doi.org/10.52202/075280-2995) and the
BFDT method (Huang, H., Ou, W. & Xiong, J., Biomed. Signal Process. Control 108, 107893 (2025)).
To cite this code, see [`CITATION.cff`](CITATION.cff) (repository: https://github.com/dev-zipida-com/kr-rPPG).

## Contact

Questions about the code: please open a GitHub issue in this repository. For access to the dataset, see
[Data access](#data-access).
