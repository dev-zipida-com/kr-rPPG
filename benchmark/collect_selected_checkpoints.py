#!/usr/bin/env python3
"""Collect validation-selection records without selecting or rescoring models."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

from common import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    rows = []
    for model in args.models:
        for fold in range(5):
            path = args.runs / f'{model}_fold{fold}/selected_checkpoint.json'
            record = json.loads(path.read_text())
            if (record['model'] != model or record['fold'] != fold or
                    record['policy'] != 'minimum_validation_loss_first_tie' or
                    record['test_results_used_for_selection'] is not False):
                raise ValueError(f'Unexpected checkpoint-selection record: {path}')
            if not 0 <= record['epoch'] < 10:
                raise ValueError(f'Unexpected training epoch: {path}')
            if not Path(record['checkpoint']).is_file():
                raise ValueError(f'Selected checkpoint is absent: {path}')
            rows.append({'model': model, 'fold': fold, 'epoch_zero_based': record['epoch'],
                         'validation_loss': record['validation_loss'], 'policy': record['policy'],
                         'checkpoint': record['checkpoint'], 'checkpoint_sha256': record['checkpoint_sha256'],
                         'selection_record': str(path),
                         'selection_record_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'expanded_config_sha256': record['expanded_config_sha256'],
                         'test_results_used_for_selection': False})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'selected_checkpoints': len(rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
