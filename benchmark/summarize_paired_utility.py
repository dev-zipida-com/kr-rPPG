#!/usr/bin/env python3
"""Aggregate the prespecified 2x2 internal and three external utility controls."""
from pathlib import Path
import argparse
import csv
import hashlib
import json

import numpy as np
import pandas as pd

MODELS = ['physnet', 'physformer']
EXTERNAL = ['pure', 'ubfc_rppg', 'mmpd']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_result(path, reference, internal, require_only_two=False):
    frame = pd.read_csv(path, dtype={'model': str, 'fold': str, 'session': str, 'participant_key': str, 'task': str})
    provenance = json.loads(path.with_suffix('.provenance.json').read_text())
    if provenance['partial_monitoring'] or provenance['model_specific_label_hr_used']:
        raise ValueError(f'Incomplete or model-dependent references: {path}')
    if provenance['reference_sha256'] != sha(reference):
        raise ValueError(f'Unexpected reference: {path}')
    if require_only_two and set(frame.model) != set(MODELS):
        raise ValueError(f'Unexpected raw control model set: {path}')
    frame = frame[frame.model.isin(MODELS)].copy()
    ref = pd.read_csv(reference, dtype={'session': str, 'fold': str, 'participant_key': str}).set_index('session')
    if frame.duplicated(['model', 'fold', 'session']).any():
        raise ValueError(f'Duplicate prediction: {path}')
    for model in MODELS:
        selected = frame[frame.model == model]
        if internal:
            if len(selected) != len(ref) or set(selected.session) != set(ref.index):
                raise ValueError(f'Incomplete OOF cohort: {path}/{model}')
            if not selected.fold.to_numpy().tolist() == ref.loc[selected.session, 'fold'].to_numpy().tolist():
                raise ValueError(f'Incorrect OOF fold: {path}/{model}')
        else:
            if set(selected.fold) != set('01234'):
                raise ValueError(f'Missing external fold: {path}/{model}')
            for fold, group in selected.groupby('fold'):
                if len(group) != len(ref) or set(group.session) != set(ref.index):
                    raise ValueError(f'Incomplete external cohort: {path}/{model}/{fold}')
    if not np.allclose(frame.reference_hr, ref.loc[frame.session, 'reference_hr'], rtol=0, atol=1e-10):
        raise ValueError(f'Noncanonical reference HR: {path}')
    if not np.isfinite(frame[['pred_hr', 'reference_hr', 'abs_error']]).all().all():
        raise ValueError(f'Non-finite value: {path}')
    if not np.allclose(frame.abs_error, (frame.pred_hr - frame.reference_hr).abs(), rtol=0, atol=1e-10):
        raise ValueError(f'Incorrect absolute error: {path}')
    return frame, {'path': str(path), 'sha256': sha(path), 'reference_sha256': sha(reference),
                   'provenance_sha256': sha(path.with_suffix('.provenance.json'))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-results', type=Path, required=True)
    parser.add_argument('--raw-results', type=Path, required=True)
    parser.add_argument('--raw-to-release', type=Path, required=True)
    parser.add_argument('--release-to-raw', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--external-reference-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources, frames, cells = {}, {}, []
    definitions = [
        ('released', 'released', args.release_results / 'kr_kfold.csv'),
        ('raw', 'raw', args.raw_results / 'kr_kfold.csv'),
        ('raw', 'released', args.raw_to_release),
        ('released', 'raw', args.release_to_raw),
    ]
    for training, testing, path in definitions:
        key = f'{training}_to_{testing}'
        frames[key], sources[key] = read_result(path, args.reference, True, require_only_two=key != 'released_to_released')
        for model in MODELS:
            group = frames[key][frames[key].model == model]
            participant_mae = group.groupby('participant_key').abs_error.mean()
            cells.append({'model': model, 'training_videos': training, 'test_dataset': 'kr_rppg',
                          'test_videos': testing, 'mae': float(participant_mae.mean()),
                          'n_participants': len(participant_mae), 'n_sessions': len(group),
                          'aggregation': 'mean of participant mean absolute errors'})
    for dataset in EXTERNAL:
        reference = args.external_reference_dir / f'{dataset}_reference_hr.csv'
        for training, results in [('released', args.release_results), ('raw', args.raw_results)]:
            key = f'{training}_to_{dataset}'
            frames[key], sources[key] = read_result(results / f'kr_to_{dataset}.csv', reference, False, require_only_two=training == 'raw')
            for model in MODELS:
                group = frames[key][frames[key].model == model]
                fold_mae = group.groupby('fold').abs_error.mean()
                cells.append({'model': model, 'training_videos': training, 'test_dataset': dataset,
                              'test_videos': 'original_external', 'mae': float(fold_mae.mean()),
                              'n_folds': len(fold_mae), 'n_sessions_per_fold': int(group.groupby('fold').size().iloc[0]),
                              'fold_mae': fold_mae.to_dict(),
                              'aggregation': 'mean of five fold-wise session mean absolute errors'})
    paired_rows, contrasts = [], []
    pair_definitions = [
        ('kr_raw_test', 'raw_to_raw', 'released_to_raw', True),
        ('kr_release_test', 'raw_to_released', 'released_to_released', True),
        ('kr_matched_video_arms', 'raw_to_raw', 'released_to_released', True),
    ] + [(dataset, f'raw_to_{dataset}', f'released_to_{dataset}', False) for dataset in EXTERNAL]
    for name, raw_key, release_key, internal in pair_definitions:
        left, right = frames[raw_key], frames[release_key]
        pair = left.merge(right, on=['model', 'fold', 'session'], suffixes=('_raw_arm', '_released_arm'), validate='one_to_one', how='outer', indicator=True)
        if not pair['_merge'].eq('both').all():
            raise ValueError(f'Unpaired utility rows: {name}')
        if not np.allclose(pair.reference_hr_raw_arm, pair.reference_hr_released_arm, rtol=0, atol=1e-10):
            raise ValueError(f'Unpaired reference HR: {name}')
        pair['delta_ae'] = pair.abs_error_released_arm - pair.abs_error_raw_arm
        for model, group in pair.groupby('model'):
            if internal:
                delta = group.groupby('participant_key_raw_arm').delta_ae.mean().mean()
            else:
                delta = group.groupby('fold').delta_ae.mean().mean()
            contrasts.append({'comparison': name, 'model': model, 'delta_mae_released_minus_raw': float(delta),
                              'n_paired_rows': len(group), 'raw_arm': raw_key, 'released_arm': release_key})
        for row in pair.to_dict('records'):
            paired_rows.append({'comparison': name, 'model': row['model'], 'fold': row['fold'], 'session': row['session'],
                                'participant_key': row['participant_key_raw_arm'],
                                'reference_hr': row['reference_hr_raw_arm'],
                                'raw_arm_ae': row['abs_error_raw_arm'], 'released_arm_ae': row['abs_error_released_arm'],
                                'delta_ae_released_minus_raw': row['delta_ae']})
    if len(cells) != 20:
        raise ValueError('Table 10 must contain all 20 numeric cells')
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(paired_rows).to_csv(args.output / 'paired_utility_results.csv', index=False)
    pd.DataFrame(cells).drop(columns=['fold_mae']).to_csv(args.output / 'table10_cells.csv', index=False)
    report = {'schema_version': 'release-basis-paired-utility-v1', 'script_sha256': sha(__file__),
              'sources': sources, 'cells': cells, 'paired_contrasts': contrasts,
              'all_internal_pairs_same_fold_reference_and_session': True,
              'same_domain_oof_reused_without_duplicate_inference': True,
              'confidence_intervals': 'Not introduced; Table 10 reports the prespecified mean MAE only',
              'paired_results_sha256': sha(args.output / 'paired_utility_results.csv')}
    (args.output / 'table10_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'table10_cells': len(cells), 'paired_rows': len(paired_rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
