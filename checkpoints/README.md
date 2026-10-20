# Checkpoints

**No trained weights are distributed with this repository.** Train the KR-rPPG models yourself with the configs
in `benchmark/configs/` (main README, steps 2 and 5). The evaluation configs name the files they expect below
`$KR_RPPG_CHECKPOINTS`: `kr_5fold/<model>/KFold_<model>_fold<k>_Epoch<e>.pth` for the k-fold models and
`mmpd_trained/<model>/...` for the MMPD-trained models.

## Official rPPG-Toolbox weights

The PURE -> KR-rPPG and UBFC-rPPG -> KR-rPPG rows of Table 8 score the official rPPG-Toolbox weights. They are
third-party files and are not re-hosted here: download `final_model_release/` from
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) at commit `b7500b848f84ad7f86e277b4612563b69f4f88f9`,
point `RPPG_TOOLBOX_WEIGHTS` to that folder and check the files against `checkpoints_manifest.csv`:

```bash
python checkpoints/verify_checkpoints.py --upstream "$RPPG_TOOLBOX_WEIGHTS" --load
```

| Column | Meaning |
|---|---|
| `file` | `upstream:final_model_release/<name>`; `<name>` is the file inside `$RPPG_TOOLBOX_WEIGHTS` |
| `architecture` | PhysNet, PhysFormer, DeepPhys, TS-CAN or EfficientPhys |
| `training_data` | PURE or UBFC-rPPG |
| `bytes`, `sha256`, `n_params` | file size, SHA-256 hash, number of parameters |

Files listed in the manifest but absent are reported as missing, not as errors. These weights are provided by
rPPG-Toolbox under its licence (Responsible AI Source Code License v1.1, including the use restrictions of
Section 3.2) and were trained on research-only datasets; check the upstream terms before using them.

## Loading weights safely

Checkpoints are PyTorch `state_dict`s; keys may carry the `module.` prefix of `DataParallel`
(`verify_checkpoints.py` strips it). Load weights only with
`torch.load(path, map_location="cpu", weights_only=True)`: `.pth` files are pickles and must not be loaded from
untrusted sources without `weights_only=True`. Models you train with the RAIL-licensed toolbox code may be
derivative works of it; see [`third_party/rPPG-Toolbox/USE_RESTRICTIONS.md`](../third_party/rPPG-Toolbox/USE_RESTRICTIONS.md).
