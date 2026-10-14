#!/usr/bin/env python3
"""Run rPPG-Toolbox (KR-rPPG adaptation) on one or more benchmark configs, one after another.

    python benchmark/run.py kr_kfold/train --gpu 0                 # train the 25 fold models
    python benchmark/run.py kr_kfold/eval                          # out-of-fold evaluation (Table 8, Fig. 3)
    python benchmark/run.py kr_to_pure/physnet_fold0.yaml --gpu 1  # a single run

Logs go to $KR_RPPG_WORK/logs/<config path>.log; the toolbox writes checkpoints and test-output pickles
under the LOG.PATH of each config ($KR_RPPG_WORK/runs/...). The run stops at the first failing config.
Set RPPG_SEED (default 100), RPPG_DATALOADER_WORKERS and RPPG_PREFETCH_FACTOR to control the toolbox.
PhysFormer configs need the upstream loss file fetched by tools/fetch_physformer_loss.py; without it the run
stops before starting any config.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import yaml

from common import CONFIGS, TOOLBOX, expand_configs, import_toolbox, work_dir


def physformer_configs(paths: List[Path]) -> List[Path]:
    """Configs that build the PhysFormer trainer, which needs the separately fetched upstream loss file."""
    out = []
    for path in paths:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if (cfg.get("MODEL") or {}).get("NAME") == "PhysFormer" and cfg.get("TOOLBOX_MODE") in (
                "train_and_test", "only_test"):
            out.append(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", help="config files or folders under benchmark/configs/")
    parser.add_argument("--gpu", default=None, help="value for CUDA_VISIBLE_DEVICES")
    parser.add_argument("--dry-run", action="store_true", help="list the commands only")
    args = parser.parse_args(argv)

    configs = expand_configs(args.configs)
    if not args.dry_run and physformer_configs(configs):
        import_toolbox()
        from neural_methods.loss.PhysFormerLossAdapter import UpstreamLossMissing, check_upstream_loss

        try:
            check_upstream_loss()
        except UpstreamLossMissing as exc:
            print(f"[run] {exc}", file=sys.stderr)
            return 1
    logs = work_dir() / "logs"
    env = dict(os.environ)
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for path in configs:
        path = path.resolve()
        rel = path.relative_to(CONFIGS) if CONFIGS in path.parents else path.name
        log = logs / f"{str(rel)[:-len('.yaml')]}.log"
        cmd = [sys.executable, "main.py", "--config_file", str(path)]
        print(f"[run] {rel} -> {log}", flush=True)
        if args.dry_run:
            continue
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with open(log, "w", encoding="utf-8") as handle:
            result = subprocess.run(cmd, cwd=TOOLBOX, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        print(f"[run] {rel}: exit {result.returncode} after {(time.time() - start) / 60:.1f} min", flush=True)
        if result.returncode != 0:
            print(f"[run] failed; see {log}", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
