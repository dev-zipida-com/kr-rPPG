# rPPG-Toolbox (KR-rPPG adaptation)

Subset of [rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) (commit `b7500b8`) with the changes
needed for the KR-rPPG benchmark. Licence: Responsible AI Source Code License v1.1 ([`LICENSE`](LICENSE));
use restrictions: [`USE_RESTRICTIONS.md`](USE_RESTRICTIONS.md); list of changes: [`NOTICE.md`](NOTICE.md)
and [`CHANGES/`](CHANGES).

Run from this folder (the toolbox resolves some paths relative to it):

```bash
cd third_party/rPPG-Toolbox
python main.py --config_file ../../benchmark/configs/kr_kfold/eval/fold0_physnet.yaml
```

or use `benchmark/run.py`, which does this for you. See the repository README for the full workflow.
For the original toolbox documentation, see the upstream repository.

`neural_methods/loss/PhysFormerLossComputer.py` is not distributed in this folder (its header states GPL-3.0
ancestry, see `NOTICE.md`). Before training or evaluating PhysFormer, fetch it from the repository root with
`python tools/fetch_physformer_loss.py` (or keep it outside the repository with `--dest` and
`KR_RPPG_PHYSFORMER_LOSS`, see the repository README); `neural_methods/loss/PhysFormerLossAdapter.py` applies the
KR-rPPG change to it.
