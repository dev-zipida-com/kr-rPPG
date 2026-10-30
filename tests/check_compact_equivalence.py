"""One small numerical check: whole-session normalization equals lazy clips.

Run directly with Python; does not require pytest, a GPU, or participant videos.
"""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'third_party' / 'rPPG-Toolbox'))
from dataset.data_loader.BaseLoader import BaseLoader
from dataset.data_loader.CompactCache import CompactCache
from dataset.data_loader.KoreanrPPGLoader import KoreanrPPGLoader


def check(folder):
    rng = np.random.default_rng(100)
    session = 'p001_T00'
    path = folder / session
    path.mkdir()
    n = 517
    bvp = np.sin(np.arange(n, dtype=np.float64) * 0.21) + rng.normal(0, 0.1, n)
    np.save(path / 'bvp.npy', bvp)
    stats = {'session': session, 'participant_key': 'p001', 'task': 'T00', 'frames': n,
             'skip_frames': 180, 'roi': [0, 0, 128, 128], 'sizes': {},
             'label': {'mean': float(bvp.mean()), 'std': float(bvp.std()), 'diff_std': float(np.diff(bvp).std())}}
    cases = []
    for size in (36, 72, 128):
        frames = rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8)
        np.save(path / f'crop_{size}.npy', frames)
        values = frames.astype(np.float32)
        differences = (values[1:] - values[:-1]) / (values[1:] + values[:-1] + np.float32(1e-7))
        stats['sizes'][str(size)] = {'mean': float(values.mean()), 'std': float(values.std()),
                                   'diff_std': float(differences.std())}
        (path / 'stats.json').write_text(json.dumps(stats))
        expected_diff = BaseLoader.diff_normalize_data(values)
        expected_std = BaseLoader.standardized_data(values)
        expected_label = BaseLoader.diff_normalize_label(bvp).astype(np.float32)
        reader = CompactCache(folder, size)
        for start, stop in ((0, 128), (128, 256), (n - 128, n)):
            actual, label = reader.clip(session, start, stop, ['DiffNormalized', 'Standardized'], 'DiffNormalized')
            expected = np.concatenate([expected_diff[start:stop], expected_std[start:stop]], axis=-1)
            error = float(np.max(np.abs(actual - expected)))
            label_error = float(np.max(np.abs(label - expected_label[start:stop])))
            if not np.array_equal(actual, expected) or not np.array_equal(label, expected_label[start:stop]):
                raise AssertionError((size, start, stop, error, label_error))
            cases.append({'size': size, 'start': start, 'stop': stop, 'input_max_abs_diff': error,
                          'label_max_abs_diff': label_error})
        del reader, frames, values, differences, expected_diff, expected_std, actual, expected
    with (folder / 'manifest.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['session', 'participant_key', 'task', 'frames', 'start_frame', 'common_eval_frames'])
        w.writerow([session, 'p001', 'T00', n, 180, min(n // c * c for c in (128, 160, 180))])
    pre = SimpleNamespace(RESIZE=SimpleNamespace(H=72, W=72), DO_CHUNK=True, CHUNK_LENGTH=128,
                          USE_PSUEDO_PPG_LABEL=False, DATA_AUG=['None'], DATA_TYPE=['DiffNormalized'], LABEL_TYPE='DiffNormalized')
    cfg = SimpleNamespace(DO_PREPROCESS=False, DATA_FORMAT='NCDHW', BEGIN=0.0, END=1.0,
                          PREPROCESS=pre, KFOLD_CSV='', KFOLD_ID=-1, KFOLD_TEST_FOLD=False)
    old = os.environ.get('KR_RPPG_COMPACT_ROOT')
    os.environ['KR_RPPG_COMPACT_ROOT'] = str(folder)
    try:
        loader = KoreanrPPGLoader('test', str(folder), cfg)
        x, y, name, index = loader[1]
        assert len(loader) == 4 and x.shape == (3, 128, 72, 72) and y.shape == (128,)
        assert name == session and index == '1'
    finally:
        if old is None:
            os.environ.pop('KR_RPPG_COMPACT_ROOT', None)
        else:
            os.environ['KR_RPPG_COMPACT_ROOT'] = old
    return {'status': 'passed', 'synthetic_sessions': 1, 'numerical_cases': cases,
            'loader_virtual_clips': 4, 'gpu_used': False, 'max_absolute_difference': 0.0}


def check_existing(folder, session):
    """Read one builder-produced session; compare only three intervals per size."""
    cases = []
    for size in (36, 72, 128):
        reader = CompactCache(folder, size)
        frames, bvp, _ = reader.session(session)
        n = len(frames)
        values = frames.astype(np.float32)
        diff = BaseLoader.diff_normalize_data(values)
        standardized = BaseLoader.standardized_data(values)
        labels = BaseLoader.diff_normalize_label(bvp).astype(np.float32)
        for start, stop in ((0, 128), (n // 2, n // 2 + 128), (n - 128, n)):
            actual, label = reader.clip(session, start, stop, ['DiffNormalized', 'Standardized'], 'DiffNormalized')
            expected = np.concatenate([diff[start:stop], standardized[start:stop]], axis=-1)
            input_error = float(np.max(np.abs(actual - expected)))
            label_error = float(np.max(np.abs(label - labels[start:stop])))
            if not np.allclose(actual, expected, rtol=1e-6, atol=1e-5) or not np.array_equal(label, labels[start:stop]):
                raise AssertionError((size, start, stop, input_error, label_error))
            cases.append({'size': size, 'start': start, 'stop': stop, 'input_max_abs_diff': input_error,
                          'label_max_abs_diff': label_error})
        del reader, frames, bvp, values, diff, standardized, actual, expected
    return {'status': 'passed', 'real_sessions': [session], 'numerical_cases': cases,
            'gpu_used': False, 'max_absolute_difference': max(c['input_max_abs_diff'] for c in cases),
            'comparison': 'float64 accumulated compact moments versus float32 full-array reduction',
            'tolerance': {'rtol': 1e-6, 'atol': 1e-5}, 'label_comparison': 'bitwise equal float32'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--compact-root', type=Path)
    p.add_argument('--session', default='p001_T00')
    args = p.parse_args()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.compact_root:
        result = check_existing(args.compact_root, args.session)
    else:
        with tempfile.TemporaryDirectory(dir=args.report.parent) as folder:
            result = check(Path(folder))
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
