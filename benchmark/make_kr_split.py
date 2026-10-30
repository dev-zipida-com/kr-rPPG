#!/usr/bin/env python3
"""Publish the deterministic participant split without private IDs or outcomes.

Each held-out fold is unchanged. The sorted non-test public IDs are split at
floor(0.9*n): first 90% train, remainder validation. Seed 100 is a training RNG
setting and does not randomize this split.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def generate(fold_path, output):
    rows = list(csv.DictReader(fold_path.open()))
    assignment = {r['participant_key']: int(r['fold']) for r in rows}
    if len(assignment) != len(rows) or set(assignment.values()) != set(range(5)):
        raise ValueError('Expected unique participants and all five folds')
    records, audit = [], []
    for fold in range(5):
        test = {p for p, f in assignment.items() if f == fold}
        pool = sorted(set(assignment) - test)
        boundary = int(0.9 * len(pool))
        train, valid = set(pool[:boundary]), set(pool[boundary:])
        if train & valid or train & test or valid & test or train | valid | test != set(assignment):
            raise AssertionError('Invalid participant partition')
        records.extend({'fold': fold, 'participant_key': p, 'split': 'train' if p in train else 'valid'} for p in pool)
        audit.append({'fold': fold, 'train_participants': len(train), 'valid_participants': len(valid),
                      'test_participants': len(test), 'overlap': 0, 'union_participants': len(assignment)})
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'train_valid_split.csv'
    with target.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'participant_key', 'split'])
        writer.writeheader()
        writer.writerows(records)
    report = {'method': 'sorted public participant IDs; first floor(0.9*n) train, remainder valid',
              'split_randomized': False, 'training_seed': 100, 'participants': len(assignment),
              'rows': len(records), 'fold_assignment_sha256': hashlib.sha256(fold_path.read_bytes()).hexdigest(),
              'split_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'folds': audit}
    (output / 'split_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold-assignment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    generate(args.fold_assignment, args.out)
