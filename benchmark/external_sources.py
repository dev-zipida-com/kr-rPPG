"""Canonical external sessions and source readers for the release-basis benchmark.

No first-six-second removal is applied to the external datasets. Source paths
and fingerprints stay in the local run directory; they are not release assets.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat, whosmat

from make_configs import UBFC_PHYS_EXCLUSION_LIST

DATASETS = {'pure': ('PURE', 30, 59), 'ubfc_rppg': ('UBFC-rPPG', 30, 42),
            'mmpd': ('MMPD', 30, 655), 'ubfc_phys': ('UBFC-PHYS', 35, 101)}
SIZES = (36, 72, 128)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def selected_fingerprint(path, block=16384):
    """Size/mtime plus first/last 16 KiB; explicitly not a full payload hash."""
    path = Path(path)
    stat = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as f:
        h.update(f.read(block))
        if stat.st_size > block:
            f.seek(max(block, stat.st_size - block))
            h.update(f.read(block))
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'selected_sha256': h.hexdigest(), 'method': 'first_last_16384_bytes_not_full_hash'}


def open_video(path):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    return cap


def raw_bvp(dataset, root, row):
    source = Path(root) / row['source']
    if dataset == 'pure':
        with (Path(root) / row['bvp_source']).open() as f:
            data = json.load(f)
        return np.asarray([r['Value']['waveform'] for r in data['/FullPackage']], dtype=np.float64)
    if dataset == 'ubfc_rppg':
        with (Path(root) / row['bvp_source']).open() as f:
            return np.asarray([float(v) for v in f.readline().split()], dtype=np.float64)
    if dataset == 'ubfc_phys':
        return np.loadtxt(Path(root) / row['bvp_source'], delimiter=',', ndmin=1).reshape(-1).astype(np.float64)
    return np.asarray(loadmat(source, variable_names=['GT_ppg'])['GT_ppg'], dtype=np.float64).reshape(-1)


def aligned_bvp(dataset, values, frames):
    values = np.asarray(values, dtype=np.float64)
    if dataset == 'ubfc_rppg':
        if len(values) != frames:
            raise ValueError(f'UBFC-rPPG video/BVP length mismatch: {frames}/{len(values)}')
        return values
    # Exact index-domain interpolation used by BaseLoader.resample_ppg.
    return np.interp(np.linspace(1, len(values), frames), np.linspace(1, len(values), len(values)), values)


def inventory(dataset, root):
    root = Path(root)
    display, fs, expected = DATASETS[dataset]
    rows = []
    if dataset == 'pure':
        for folder in sorted(root.glob('[0-9][0-9]-[0-9][0-9]')):
            name = folder.name
            pngs = sorted((folder / name).glob('*.png'))
            if not pngs:
                raise ValueError(f'No PURE images: {folder}')
            rows.append({'session': str(int(name.replace('-', ''))), 'participant_key': f'pure_s{name[:2]}',
                         'task': name[-2:], 'source': str((folder / name).relative_to(root)),
                         'bvp_source': str((folder / f'{name}.json').relative_to(root)), 'frames': len(pngs),
                         'source_fingerprint': {'ordered_png_names_sha256': hashlib.sha256('\n'.join(p.name for p in pngs).encode()).hexdigest(),
                                                'first_png': selected_fingerprint(pngs[0]), 'last_png': selected_fingerprint(pngs[-1])}})
    elif dataset in ('ubfc_rppg', 'ubfc_phys'):
        pattern = 'subject*/vid.avi' if dataset == 'ubfc_rppg' else 's*/vid_s*_T*.avi'
        for video in sorted(root.glob(pattern)):
            name = video.parent.name if dataset == 'ubfc_rppg' else video.stem.removeprefix('vid_')
            subject = video.parent.name.removeprefix('subject').removeprefix('s')
            gt = video.parent / ('ground_truth.txt' if dataset == 'ubfc_rppg' else f'bvp_{name}.csv')
            cap = open_video(video)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            header_fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            rows.append({'session': name, 'participant_key': f'{dataset}_s{subject}',
                         'task': '' if dataset == 'ubfc_rppg' else name.split('_')[-1],
                         'source': str(video.relative_to(root)), 'bvp_source': str(gt.relative_to(root)),
                         'frames': frames, 'video_header_fps': header_fps,
                         'source_fingerprint': selected_fingerprint(video)})
    else:
        for path in sorted(root.glob('subject*/*.mat')):
            shape = {name: shape for name, shape, _ in whosmat(path)}['video']
            if len(shape) != 4 or shape[-1] != 3:
                raise ValueError(f'Unsupported MMPD video layout: {path}: {shape}')
            rows.append({'session': path.stem, 'participant_key': f'mmpd_s{path.parent.name[7:]}',
                         'task': path.stem.split('_')[-1], 'source': str(path.relative_to(root)),
                         'bvp_source': str(path.relative_to(root)), 'frames': int(shape[0]),
                         'video_shape': list(shape), 'source_fingerprint': selected_fingerprint(path)})
    if len({r['session'] for r in rows}) != len(rows):
        raise ValueError(f'{dataset}: duplicate session names')
    for row in rows:
        row.update(dataset=dataset, toolbox_dataset=display, fs=fs, start_frame=0)
        row['eligible'] = not (dataset == 'ubfc_phys' and row['session'] in UBFC_PHYS_EXCLUSION_LIST)
        row['reason'] = 'canonical_public_exclusion_list' if not row['eligible'] else 'eligible'
        row['common_eval_frames'] = min(row['frames'] // c * c for c in (128, 160, 180))
        signal = raw_bvp(dataset, root, row)
        if not len(signal) or not np.isfinite(signal).all():
            raise ValueError(f"{dataset}/{row['session']}: invalid contact BVP")
        row['raw_bvp_samples'] = len(signal)
        row['raw_bvp_values_sha256'] = hashlib.sha256(signal.astype('<f8').tobytes()).hexdigest()
        if dataset != 'mmpd':
            row['bvp_file_sha256'] = sha256(root / row['bvp_source'])
        aligned_bvp(dataset, signal, row['frames'])  # validate the fixed alignment policy
        if row['common_eval_frames'] <= 0:
            raise ValueError(f"{dataset}/{row['session']}: no common model interval")
    retained = [r for r in rows if r['eligible']]
    if len(retained) != expected:
        raise ValueError(f'{dataset}: expected {expected} canonical sessions, found {len(retained)}')
    return {'schema_version': 'external-source-v1', 'dataset': dataset, 'root': str(root), 'fs': fs,
            'source_sessions': len(rows), 'eligible_sessions': len(retained),
            'excluded_sessions': len(rows) - len(retained), 'skip_frames': 0,
            'roi_policy': 'first RGB frame Haar detector; largest width face; enlargement 1.5; fixed per session',
            'bvp_policy': 'equal-length contact BVP' if dataset == 'ubfc_rppg' else 'toolbox index-domain linear interpolation to video frame count',
            'normalization_policy': 'float32 frame/difference arithmetic; float64 accumulated full-session moments',
            'legacy_precision_note': 'Legacy external resize arrays were float64; compact crops are uint8 and use the new common float32 normalization protocol.',
            'estimated_crop_bytes_by_size': {str(s): sum(r['frames'] * s * s * 3 for r in retained) for s in SIZES},
            'fingerprint_scope': 'Selected container/image bytes, source size/mtime, and complete GT values; no full raw MAT/video hashes.',
            'sessions': rows}
