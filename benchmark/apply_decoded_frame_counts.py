#!/usr/bin/env python3
"""Apply verified UBFC-Phys decode counts to a separate source inventory.

The amendment contains observed counts, not a rule to subtract frames. Source
size, selected container bytes and contact-BVP hashes bind each correction to
the inventoried recording. Local root paths and mtimes are deliberately not
compared. This command reads JSON only; the subsequent cache build must still
decode every eligible recording and reject any count mismatch.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


SCHEMA = 'ubfc-phys-decoded-counts-v1'
FINGERPRINT_METHOD = 'first_last_16384_bytes_not_full_hash'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f'{name}: expected a positive integer')
    return value


def digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{64}', value) is None:
        raise ValueError(f'{name}: expected a lowercase SHA-256 digest')


def validate_amendment(amendment):
    """Check the portable contract without reading videos or waveforms."""
    required = {'schema_version': SCHEMA, 'dataset': 'ubfc_phys',
                'eligible_sessions': 101, 'status': 'verified_from_completed_decode'}
    for key, expected in required.items():
        if amendment.get(key) != expected:
            raise ValueError(f'Amendment {key}: expected {expected!r}')
    corrections = amendment.get('corrections')
    if not isinstance(corrections, list) or not corrections:
        raise ValueError('Amendment must contain observed corrections')
    seen = set()
    for row in corrections:
        session = row.get('session')
        if not isinstance(session, str) or not session or session in seen:
            raise ValueError(f'Invalid or duplicate correction session: {session!r}')
        seen.add(session)
        source = row.get('source')
        if (not isinstance(source, str) or not source or '\\' in source
                or PurePosixPath(source).is_absolute() or '..' in PurePosixPath(source).parts
                or str(PurePosixPath(source)) != source):
            raise ValueError(f'{session}: source must be a normalized relative POSIX path')
        header = positive_integer(row.get('header_frames'), f'{session}/header_frames')
        decoded = positive_integer(row.get('decoded_frames'), f'{session}/decoded_frames')
        if decoded == header:
            raise ValueError(f'{session}: correction must change the frame count')
        if row.get('fs') != 35:
            raise ValueError(f'{session}: expected nominal fs=35')
        endpoint = min(decoded // chunk * chunk for chunk in (128, 160, 180))
        if endpoint != 6240 or row.get('common_eval_frames') != endpoint:
            raise ValueError(f'{session}: correction must preserve common_eval_frames=6240')
        fingerprint = row.get('source_fingerprint', {})
        positive_integer(fingerprint.get('size_bytes'), f'{session}/size_bytes')
        digest(fingerprint.get('selected_sha256'), f'{session}/selected_sha256')
        if fingerprint.get('method') != FINGERPRINT_METHOD:
            raise ValueError(f'{session}: unsupported selected-byte fingerprint method')
        digest(row.get('raw_bvp_values_sha256'), f'{session}/raw_bvp_values_sha256')
    return corrections


def apply_counts(inventory, amendment):
    """Return an amended copy; never mutate the immutable header inventory."""
    corrections = validate_amendment(amendment)
    if inventory.get('dataset') != 'ubfc_phys' or inventory.get('fs') != 35:
        raise ValueError('Expected a UBFC-Phys inventory with nominal fs=35')
    sessions = inventory.get('sessions', [])
    by_session = {row['session']: row for row in sessions}
    if len(by_session) != len(sessions):
        raise ValueError('Inventory contains duplicate sessions')
    eligible = [row for row in sessions if row.get('eligible') is True]
    if inventory.get('eligible_sessions') != 101 or len(eligible) != 101:
        raise ValueError('Expected exactly 101 eligible UBFC-Phys recordings')
    for correction in corrections:
        session = correction['session']
        row = by_session.get(session)
        if row is None or row.get('eligible') is not True:
            raise ValueError(f'{session}: correction is outside the eligible inventory')
        checks = {'dataset': 'ubfc_phys', 'source': correction['source'],
                  'frames': correction['header_frames'], 'fs': correction['fs'],
                  'common_eval_frames': correction['common_eval_frames'],
                  'raw_bvp_values_sha256': correction['raw_bvp_values_sha256']}
        for key, expected in checks.items():
            if row.get(key) != expected:
                raise ValueError(f'{session}/{key}: inventory does not match amendment')
        fingerprint = row.get('source_fingerprint', {})
        for key in ('size_bytes', 'selected_sha256', 'method'):
            if fingerprint.get(key) != correction['source_fingerprint'][key]:
                raise ValueError(f'{session}/source_fingerprint/{key}: inventory does not match amendment')
    result = copy.deepcopy(inventory)
    corrected = {row['session']: row for row in corrections}
    for row in result['sessions']:
        if row['session'] in corrected:
            correction = corrected[row['session']]
            row['original_header_frames'] = row['frames']
            row['frames'] = correction['decoded_frames']
            row['common_eval_frames'] = min(row['frames'] // chunk * chunk for chunk in (128, 160, 180))
            row['frame_count_amendment'] = 'Observed complete decode count; immutable header count retained separately'
    result['estimated_crop_bytes_by_size'] = {
        str(size): sum(row['frames'] * size * size * 3 for row in result['sessions'] if row['eligible'])
        for size in (36, 72, 128)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True, help='Fresh immutable header inventory')
    parser.add_argument('--amendment', type=Path, required=True, help='Verified portable decoded-count amendment')
    parser.add_argument('--out', type=Path, required=True, help='New effective inventory; never the input file')
    args = parser.parse_args()
    note_path = args.out.with_suffix('.repair_note.json')
    if args.out.resolve() in (args.inventory.resolve(), args.amendment.resolve()):
        raise ValueError('Effective inventory must have a separate output path')
    if args.out.exists() or note_path.exists():
        raise FileExistsError('Effective inventory or repair note already exists; use a fresh output path')
    inventory = json.loads(args.inventory.read_text())
    amendment = json.loads(args.amendment.read_text())
    result = apply_counts(inventory, amendment)
    provenance = {'schema_version': SCHEMA, 'original_inventory_sha256': sha256(args.inventory),
                  'amendment_sha256': sha256(args.amendment), 'helper_sha256': sha256(__file__),
                  'corrected_sessions': sorted(row['session'] for row in amendment['corrections']),
                  'eligible_sessions': 101, 'excluded_cohort_unchanged': True,
                  'common_eval_frames_unchanged': 6240,
                  'source_matching': 'relative path, header count, nominal fs, size, selected bytes and raw BVP values; root and mtime ignored',
                  'bvp_alignment': 'Interpolate the complete raw BVP to the effective decoded frame count before selecting the common interval',
                  'decode_confirmation': 'Observed amendment only; complete subsequent cache build must confirm all actual counts',
                  'table7_affected': False}
    result['decoded_frame_count_amendment'] = provenance
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    with note_path.open('x') as handle:
        handle.write(json.dumps({**provenance, 'effective_inventory_sha256': sha256(args.out)}, indent=2) + '\n')
    print(json.dumps({'effective_inventory': str(args.out), 'repair_note': str(note_path),
                      'corrected_sessions': len(amendment['corrections']), 'eligible_sessions': 101}))


if __name__ == '__main__':
    main()
