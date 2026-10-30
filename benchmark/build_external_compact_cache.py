#!/usr/bin/env python3
"""Build one external dataset's fixed-cohort shared crops after inventory review.

Run datasets sequentially. A --sessions pilot never publishes a complete
manifest. Large MAT arrays are loaded one session per worker; default workers=1.
No model is loaded and no inference or training is performed by this command.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# NumPy reads this at import; avoid synchronous huge-page compaction on this host.
# This changes allocation policy only, preserving all pixel/statistic computations.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')

import cv2
import numpy as np
from scipy.io import loadmat

from build_compact_cache import session_stats
from external_sources import aligned_bvp, open_video, raw_bvp, selected_fingerprint, sha256

REPO = Path(__file__).resolve().parents[1]
CASCADE = REPO / 'third_party/rPPG-Toolbox/dataset/haarcascade_frontalface_default.xml'


def first_frame_roi(rgb):
    """Match BaseLoader's RGB Haar call and its fixed enlargement/cropping rules."""
    detector = cv2.CascadeClassifier(str(CASCADE))
    if detector.empty():
        raise ValueError(f'Missing Haar cascade: {CASCADE}')
    zones = detector.detectMultiScale(rgb[:, :, :3].astype(np.uint8))
    # Preserve the toolbox no-detection fallback, including its H/W convention,
    # rather than silently changing this benchmark's legacy ROI definition.
    box = [0, 0, rgb.shape[0], rgb.shape[1]] if len(zones) == 0 else zones[np.argmax(zones[:, 2])].copy()
    box[0] = max(0, box[0] - 0.25 * box[2])
    box[1] = max(0, box[1] - 0.25 * box[3])
    box[2] = 1.5 * box[2]
    box[3] = 1.5 * box[3]
    x, y, w, h = np.asarray(box, dtype=int)
    roi = [max(0, int(x)), max(0, int(y)), min(int(x + w), rgb.shape[1]), min(int(y + h), rgb.shape[0])]
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise ValueError('First-frame Haar ROI is empty')
    return roi, int(len(zones))


def frame_stream(dataset, root, row):
    source = Path(root) / row['source']
    if dataset == 'pure':
        paths = sorted(source.glob('*.png'))
        names_hash = hashlib.sha256('\n'.join(p.name for p in paths).encode()).hexdigest()
        expected = row['source_fingerprint']
        if len(paths) != row['frames'] or names_hash != expected['ordered_png_names_sha256'] or selected_fingerprint(paths[0]) != expected['first_png'] or selected_fingerprint(paths[-1]) != expected['last_png']:
            raise ValueError('PURE image inventory changed')
        for path in paths:
            frame = cv2.imread(str(path.resolve()))
            if frame is None:
                raise ValueError(f'Cannot read {path}')
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    elif dataset == 'mmpd':
        if selected_fingerprint(source) != row['source_fingerprint']:
            raise ValueError('MMPD source fingerprint changed')
        frames = loadmat(source, variable_names=['video'])['video']
        if tuple(frames.shape) != tuple(row['video_shape']):
            raise ValueError('MMPD video dimensions changed')
        for frame in frames:
            yield np.round(frame * 255).astype(np.uint8)
    else:
        if selected_fingerprint(source) != row['source_fingerprint']:
            raise ValueError('AVI source fingerprint changed')
        cap = open_video(source)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()


def build_one(job):
    dataset, root, output, row, provenance, requested_sizes = job
    cv2.setNumThreads(1)
    started = time.time()
    dest = Path(output) / row['session']
    complete = dest / 'stats.json'
    existing = {}
    if complete.exists():
        existing = json.loads(complete.read_text())
        if existing.get('cache_provenance') != provenance or existing['frames'] != row['frames']:
            raise ValueError(f'Existing external cache provenance differs: {dest}')
        if all(str(s) in existing['sizes'] and (dest / f'crop_{s}.npy').is_file() for s in requested_sizes):
            return {'session': row['session'], 'status': 'reused', 'frames': row['frames']}
    sizes = [s for s in requested_sizes if str(s) not in existing.get('sizes', {})]
    if not sizes:
        raise ValueError(f'{dest}: statistics exist but a crop file is missing')
    dest.mkdir(parents=True, exist_ok=True)
    wave = raw_bvp(dataset, root, row)
    if hashlib.sha256(wave.astype('<f8').tobytes()).hexdigest() != row['raw_bvp_values_sha256']:
        raise ValueError('Contact BVP values changed after inventory')
    wave = aligned_bvp(dataset, wave, row['frames'])
    n = row['frames']
    arrays = {s: np.lib.format.open_memmap(dest / f'crop_{s}.npy', mode='w+', dtype=np.uint8, shape=(n, s, s, 3)) for s in sizes}
    count, roi, detections = 0, None, None
    for rgb in frame_stream(dataset, root, row):
        if count >= n:
            raise ValueError(f"{row['session']}: decoded more frames than inventoried")
        if roi is None:
            roi, detections = first_frame_roi(rgb)
        x1, y1, x2, y2 = roi
        face = rgb[y1:y2, x1:x2]
        for size, array in arrays.items():
            array[count] = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)
        count += 1
    if count != n:
        raise ValueError(f"{row['session']}: decoded/header frame mismatch {count}/{n}")
    for array in arrays.values():
        array.flush()
    np.save(dest / 'bvp.npy', wave.astype(np.float64))
    stats = {'schema_version': 'compact-v1', 'dataset': dataset, 'session': row['session'],
             'participant_key': row['participant_key'], 'task': row['task'], 'frames': n, 'fs': row['fs'],
             'skip_frames': 0, 'common_eval_frames': row['common_eval_frames'], 'roi': roi,
             'first_frame_face_detections': detections, 'no_face_fallback_used': detections == 0,
             'roi_policy': 'fixed first-frame RGB Haar; 1.5 enlargement; toolbox fallback H/W convention',
             'sizes': {**existing.get('sizes', {}), **{str(s): session_stats(a) for s, a in arrays.items()}},
             'label': {'mean': float(wave.mean()), 'std': float(wave.std()), 'diff_std': float(np.diff(wave).std())},
             'source': row['source'], 'raw_bvp_values_sha256': row['raw_bvp_values_sha256'],
             'cache_provenance': provenance,
             'moment_precision': 'float32 frame/difference arithmetic; float64 accumulated population moments',
             'seconds': round(time.time() - started, 3)}
    if stats['label']['diff_std'] <= 0:
        raise ValueError('Degenerate contact BVP normalization')
    temp = complete.with_suffix('.json.tmp')
    temp.write_text(json.dumps(stats, indent=2) + '\n'); temp.replace(complete)
    return {'session': row['session'], 'status': 'created', 'frames': n, 'seconds': stats['seconds'],
            'no_face_fallback_used': stats['no_face_fallback_used']}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='One dataset directory below RPPG_EXTERNAL_COMPACT_ROOT')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--sizes', type=int, nargs='+', choices=(36, 72, 128), default=[36, 72, 128])
    p.add_argument('--sessions', nargs='*')
    args = p.parse_args()
    data = json.loads(args.inventory.read_text())
    rows = [r for r in data['sessions'] if r['eligible'] and (not args.sessions or r['session'] in args.sessions)]
    if not rows:
        raise ValueError('No matching canonical external sessions')
    if args.sessions and set(args.sessions) != {r['session'] for r in rows}:
        raise ValueError('Requested pilot sessions are outside the canonical cohort')
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = {'source_inventory_sha256': sha256(args.inventory), 'builder_sha256': sha256(Path(__file__)),
                  'source_reader_sha256': sha256(Path(__file__).with_name('external_sources.py')),
                  'moment_helper_sha256': sha256(Path(__file__).with_name('build_compact_cache.py')),
                  'haar_cascade_sha256': sha256(CASCADE)}
    required = sum(r['frames'] * s * s * 3 for r in rows for s in args.sizes if not (args.out / r['session'] / f'crop_{s}.npy').exists())
    if shutil.disk_usage(args.out).free < required + 10 * 2**30:
        raise ValueError(f'Insufficient disk: require {required / 2**30:.2f} GiB plus 10 GiB reserve')
    started = time.time(); results = []; errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(build_one, (data['dataset'], data['root'], str(args.out), row, provenance, args.sizes)): row['session'] for row in rows}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result(); results.append(result)
            except Exception as exc:
                result = {'session': futures[future], 'status': 'error', 'error': str(exc)}; errors.append(result)
            print(json.dumps({'done': i, 'total': len(rows), **result}), flush=True)
    report = {'dataset': data['dataset'], 'requested_sessions': len(rows), 'completed_sessions': len(results),
              'errors': errors, 'wall_seconds': round(time.time() - started, 3), 'workers': args.workers,
              'sizes': args.sizes, 'cache_provenance': provenance}
    (args.out / ('build_report_pilot.json' if args.sessions else 'build_report_all.json')).write_text(json.dumps(report, indent=2) + '\n')
    if errors:
        raise SystemExit(1)
    if not args.sessions:
        fields = ['dataset', 'session', 'participant_key', 'task', 'fs', 'frames', 'start_frame', 'common_eval_frames']
        with (args.out / 'manifest.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
        metadata = {'schema_version': 'external-compact-v1', 'dataset': data['dataset'], 'fs': data['fs'],
                    'sessions': sorted(r['session'] for r in rows),
                    'excluded_sessions': sorted(r['session'] for r in data['sessions'] if not r['eligible']),
                    'cache_provenance': provenance, 'source_inventory': str(args.inventory)}
        (args.out / 'cache_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
