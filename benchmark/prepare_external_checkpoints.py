#!/usr/bin/env python3
"""Verify the 15 external checkpoints and optionally create their configured aliases."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re

import yaml

MODELS = {'physnet', 'physformer', 'deepphys', 'tscan', 'efficientphys'}
SOURCES = {'PURE': 'pure', 'UBFC-rPPG': 'ubfc_rppg', 'MMPD': 'mmpd'}
ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def manifest_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        rows = [row for row in csv.DictReader(handle)
                if row['training_dataset'] != 'kr_rppg_release']
    expected = {(source, model) for source in SOURCES.values() for model in MODELS}
    if len(rows) != len(expected) or {(r['training_dataset'], r['model']) for r in rows} != expected:
        raise ValueError('Manifest must contain exactly three external datasets by five models')
    for row in rows:
        alias = PurePosixPath(row['alias'])
        if alias.is_absolute() or '..' in alias.parts or str(alias) != row['alias']:
            raise ValueError(f"Invalid relative checkpoint alias: {row['alias']}")
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError(f"Invalid SHA-256: {row['alias']}")
    if len({row['alias'] for row in rows}) != len(rows):
        raise ValueError('Duplicate checkpoint aliases')
    return rows


def legacy_sources(path, rows):
    expected = {(row['training_dataset'], Path(row['alias']).name): row['alias'] for row in rows}
    result = {}
    entries = json.loads(path.read_text())['external_checkpoint_inventory']
    for entry in entries:
        if entry['training_dataset'] == 'UBFC-Phys':
            continue
        source = SOURCES.get(entry['training_dataset'], entry['training_dataset'])
        checkpoint = Path(entry['checkpoint']).expanduser()
        key = (source, checkpoint.name)
        if key not in expected or expected[key] in result:
            raise ValueError(f'Unexpected or duplicate legacy checkpoint: {key}')
        if not checkpoint.is_absolute():
            checkpoint = path.parent / checkpoint
        result[expected[key]] = checkpoint
    if set(result) != {row['alias'] for row in rows}:
        raise ValueError('Legacy inventory does not provide all 15 manifest checkpoints')
    return result


def prepare(args):
    rows = manifest_rows(args.manifest)
    checkpoint_root = args.checkpoint_root.expanduser().absolute()
    sources = legacy_sources(args.inventory_json, rows) if args.inventory_json else {}
    records = []
    for row in rows:
        source, model = row['training_dataset'], row['model']
        config_rel = (f'mmpd_to_kr/eval/{model}.yaml' if source == 'mmpd'
                      else f'{source}_to_kr/{model}.yaml')
        config = ROOT / 'benchmark/configs' / config_rel
        cfg = yaml.safe_load(config.read_text())
        config_alias = cfg['INFERENCE']['MODEL_PATH'].replace('${RPPG_TOOLBOX_WEIGHTS}/', 'upstream/')
        config_alias = config_alias.replace('${KR_RPPG_CHECKPOINTS}/', '')
        if config_alias != row['alias']:
            raise ValueError(f'Manifest/config alias mismatch: {source}/{model}')
        alias = checkpoint_root / row['alias']
        checkpoint = sources.get(row['alias'], (args.source_root or checkpoint_root) / row['alias'])
        checkpoint = checkpoint.expanduser().absolute()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing checkpoint ({row['availability']}): {checkpoint}")
        observed = sha256(checkpoint)
        if observed != row['sha256']:
            raise ValueError(f"SHA-256 mismatch: {checkpoint}; expected {row['sha256']}, got {observed}")
        if args.link and (alias.exists() or alias.is_symlink()) and alias.resolve() != checkpoint.resolve():
            raise ValueError(f'Existing alias has a different source: {alias}')
        records.append({'training_dataset': source, 'model': model, 'checkpoint': str(checkpoint),
                        'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': observed,
                        'planned_alias': str(alias), 'alias_present': alias.is_file(),
                        'selected_epoch': int(row['epoch_zero_based']) if row['epoch_zero_based'] else None,
                        'availability': row['availability'], 'selection_scope': row['selection'],
                        'release_test_config': str(config), 'release_test_config_sha256': sha256(config),
                        'model_config': cfg['MODEL'], 'test_preprocessing': cfg['TEST']['DATA']['PREPROCESS'],
                        'test_fs': cfg['TEST']['DATA']['FS']})
    if args.link:
        for record in records:
            alias, checkpoint = Path(record['planned_alias']), Path(record['checkpoint'])
            alias.parent.mkdir(parents=True, exist_ok=True)
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(os.path.relpath(checkpoint, alias.parent))
            record['alias_present'] = alias.is_file()
    result = {'schema_version': 'external-checkpoints-v2', 'count': len(records),
              'source_manifest_sha256': sha256(args.manifest),
              'weights_deserialized': False, 'training_executed': False,
              'environment': {'KR_RPPG_CHECKPOINTS': str(checkpoint_root),
                              'RPPG_TOOLBOX_WEIGHTS': str(checkpoint_root / 'upstream')},
              'checkpoints': records}
    if args.inventory_json:
        result['source_inventory_sha256'] = sha256(args.inventory_json)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(args.out)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'checkpoints/evaluated_checkpoints.csv',
                        help='CSV with aliases and expected SHA-256 values (default: evaluated_checkpoints.csv)')
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--source-root', type=Path, help='Optional source directory with the manifest alias layout')
    inputs.add_argument('--inventory-json', type=Path, help='Optional legacy inventory of source checkpoint paths')
    parser.add_argument('--checkpoint-root', type=Path, required=True, help='Configured checkpoint directory')
    parser.add_argument('--out', type=Path, required=True, help='Verified checkpoint record JSON')
    parser.add_argument('--link', action='store_true', help='Create relative symlinks after every checkpoint passes')
    args = parser.parse_args()
    if args.link and not (args.source_root or args.inventory_json):
        parser.error('--link requires --source-root or --inventory-json')
    try:
        result = prepare(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps({'count': result['count'], 'manifest': str(args.out), 'links_created': args.link}))


if __name__ == '__main__':
    main()
