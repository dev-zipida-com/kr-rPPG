#!/usr/bin/env python3
"""Read source headers/GT and write a fixed cohort; never decode full videos or build caches."""
import argparse
import csv
import json
from pathlib import Path

from external_sources import DATASETS, inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = inventory(args.dataset, args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f'{args.dataset}_source_inventory.json').write_text(json.dumps(data, indent=2) + '\n')
    fields = ['dataset', 'session', 'participant_key', 'task', 'fs', 'frames', 'start_frame', 'common_eval_frames', 'eligible', 'reason', 'source', 'bvp_source']
    with (args.out / f'{args.dataset}_sessions.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(data['sessions'])
    print(json.dumps({k: v for k, v in data.items() if k != 'sessions'}, indent=2))


if __name__ == '__main__':
    main()
