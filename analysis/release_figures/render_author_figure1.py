#!/usr/bin/env python3
"""Render the author-only BFDT illustration from a supplied demonstration video.

This does not read or modify the released clinical cohort. Apply each transform
to the complete decoded frame before taking the shared display crop. The 0.02
panel uses the exact release-builder DC rounding definition.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import json
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frame', type=int, default=467)
    parser.add_argument('--crop', type=int, nargs=4, default=[856, 136, 1000, 280],
                        metavar=('X0', 'Y0', 'X1', 'Y1'))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    code = Path(__file__).resolve().parents[2]
    reference = code / 'privacy/bfdt_transform.py'
    spec = importlib.util.spec_from_file_location('bfdt_reference', reference)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.insert(0, str(code / 'release_builder'))
    from krrppg_release.bfdt import bfdt_frame_reference
    release_operator = code / 'release_builder/krrppg_release/bfdt.py'
    cap = cv2.VideoCapture(str(args.video), cv2.CAP_FFMPEG,
                           [cv2.CAP_PROP_N_THREADS, 1])
    count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    if not 0 <= args.frame < count:
        raise ValueError('Frame index is outside the supplied video')
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError('Could not decode selected demonstration frame')
    x0, y0, x1, y1 = args.crop
    height, width = frame.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Invalid display crop')
    ratios = [1.0, .3, .1, .02]
    panels = [frame, module.apply_bfdt(frame, 8, .3),
              module.apply_bfdt(frame, 8, .1), bfdt_frame_reference(frame, 8)]
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'source_manifest.json').exists():
        raise FileExistsError('Existing illustration manifest: choose a new output directory')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                         'text.color': 'black', 'svg.fonttype': 'none'})
    fig = plt.figure(figsize=(180/25.4, 46.5/25.4), facecolor='white')
    titles = [r'$k_r = 1.00$'+'\nOriginal', r'$k_r = 0.30$'+'\n19/64 coefficients',
              r'$k_r = 0.10$'+'\n6/64 coefficients', r'$k_r = 0.02$'+'\n1/64 (DC only)']
    panel_metadata = []
    for i, (ratio, transformed, title) in enumerate(zip(ratios, panels, titles)):
        crop = transformed[y0:y1, x0:x1]
        crop_path = args.output / f'panel_kr{ratio:.2f}.png'
        cv2.imwrite(str(crop_path), crop)
        panel_metadata.append({'keep_ratio': ratio, 'coefficients_kept': max(1, int(64*ratio)),
                               'crop_png': str(crop_path), 'crop_png_sha256': sha(crop_path)})
        ax = fig.add_axes([.004 + .250*i, .008, .242, .805])
        ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), interpolation='nearest')
        ax.set_axis_off()
        ax.set_title(title, color='black', fontsize=7.5, linespacing=1.05, pad=4)
    paths = {}
    for suffix in ('png', 'jpeg', 'pdf', 'svg'):
        path = args.output / f'Figure_1_author_demonstration.{suffix}'
        kwargs = {'pil_kwargs': {'quality': 98, 'subsampling': 0}} if suffix == 'jpeg' else {}
        fig.savefig(path, dpi=450, facecolor='white', **kwargs)
        paths[suffix] = {'path': str(path), 'sha256': sha(path)}
    plt.close(fig)
    caption = ('Figure 1. BFDT illustration using an author demonstration video. '
               'Junyoung Choi was recorded with the study equipment in the hospital measurement environment. '
               'Crops from the same composite frame (1920 × 1080 pixels) are shown after full-frame '
               'transformation at kr = 1.00, 0.30, 0.10 and 0.02. The release uses kr = 0.02; '
               'this demonstration is separate from the released participant cohort.')
    manifest = {
        'schema_version': 'author-demonstration-figure1-v1',
        'status': 'ready_for_assembly',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'asset_action': 'replaced_author_demonstration',
        'asset': paths['jpeg'], 'other_assets': {k:v for k,v in paths.items() if k != 'jpeg'},
        'source_video': {'path': str(args.video.resolve()), 'sha256': sha(args.video),
                         'frames': count, 'fps': fps, 'decoded_size_wh': [width, height],
                         'kind': 'user_supplied_composite_demonstration_with_webcam_and_frequency_plot'},
        'author': {'name': 'Junyoung Choi', 'identity_basis': 'user_statement',
                   'recording_context_basis': 'user_statement',
                   'recording_context': 'same study equipment and hospital measurement environment'},
        'separate_from_released_cohort': True,
        'written_consent_document_verified': False,
        'panel_keep_ratios': ratios, 'desired_caption': caption,
        'transform_provenance': {
            'frame_index_zero_based': args.frame,
            'selection_rule': 'middle frame of the 935-frame uploaded demonstration video',
            'frame_time_seconds': args.frame/fps,
            'decoded_frame_sha256': hashlib.sha256(frame.tobytes()).hexdigest(),
            'full_frame_size_wh': [width, height], 'block_size': 8,
            'block_grid_origin_xy': [0, 0], 'display_crop_xyxy': args.crop,
            'order': 'decode full composite frame -> full-frame BFDT -> identical face display crop -> display magnification',
            'resize_before_transform': False,
            'kr_0_02_operator': 'release_builder.krrppg_release.bfdt.bfdt_frame_reference; exact block sum / 64, nearest integer ties to even',
            'kr_0_30_and_0_10_operator': 'privacy.bfdt_transform.apply_bfdt; zigzag 19 and 6 coefficients, float32 DCT/IDCT, clip then uint8 truncation',
            'operator_sources': [{'path': str(p), 'sha256': sha(p)} for p in (reference, release_operator)],
            'renderer': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)},
            'panels': panel_metadata,
            'historical_embedded_figure_exact_generation_verified': False,
            'scope': 'Illustration of the operator on the supplied author demonstration; not an actual released cohort frame or quantitative privacy evidence.',
            'resolution_limitation': 'The uploaded composite is 1920x1080 and its webcam inset is already rendered. It is not a native 640x480 clinical acquisition file; apparent facial-detail loss depends on pixel scale.'
        }
    }
    (args.output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (args.output / 'caption.txt').write_text(caption + '\n')
    print(json.dumps({'status': 'ready_for_assembly', 'jpeg': paths['jpeg']['path'],
                      'source_manifest': str(args.output / 'source_manifest.json')}))


if __name__ == '__main__':
    main()
