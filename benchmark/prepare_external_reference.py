#!/usr/bin/env python3
"""Derive external common HR directly from contact BVP on pinned intervals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from common import import_toolbox
from collect_release_predictions import predict_hr
from external_sources import aligned_bvp, raw_bvp, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    data = json.loads(args.inventory.read_text())
    import_toolbox()
    rows = []
    for item in data['sessions']:
        if not item['eligible']:
            continue
        raw = raw_bvp(data['dataset'], data['root'], item)
        if hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest() != item['raw_bvp_values_sha256']:
            raise ValueError(f"Contact GT changed: {item['session']}")
        aligned = aligned_bvp(data['dataset'], raw, item['frames'])
        length = item['common_eval_frames']
        rows.append({'session': item['session'], 'common_eval_frames': length,
                     'reference_hr': predict_hr(aligned[:length], data['fs'], False), 'fs': data['fs'],
                     'participant_key': item['participant_key'], 'task': item['task']})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    provenance = {'dataset': data['dataset'], 'sessions': len(rows), 'input': 'raw external contact BVP',
                  'source_inventory_sha256': sha256(args.inventory), 'output_sha256': sha256(args.out),
                  'reference_generator_sha256': sha256(__file__),
                  'shared_hr_function_sha256': sha256(Path(__file__).with_name('collect_release_predictions.py')),
                  'signal_interval': 'first min(floor(N/128)*128, floor(N/160)*160, floor(N/180)*180) frames; no initial skip',
                  'postprocessing': 'float64; Tarvainen lambda100; first-order Butterworth 0.6–3.3Hz; periodogram next power of two',
                  'legacy_oof_or_model_labels_used': False, 'decode_count_confirmation': 'checked during subsequent cache build'}
    args.out.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'dataset': data['dataset'], 'sessions': len(rows), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
