"""Optional shared external test cache; no model-specific session subsets.

Set RPPG_EXTERNAL_COMPACT_ROOT to the parent of pure/, ubfc_rppg/, mmpd/,
ubfc_phys/. Training and validation keep their existing source loaders.
"""
import csv
import json
import os
from pathlib import Path

import numpy as np
from .CompactCache import CompactCache

DATASETS = {'PURE': ('pure', 30, 59), 'UBFC-rPPG': ('ubfc_rppg', 30, 42),
            'MMPD': ('mmpd', 30, 655), 'UBFC-PHYS': ('ubfc_phys', 35, 101)}


class ExternalCompactMixin:
    def _init_external_compact(self, name, data_path, config_data):
        self.external_compact_cache = None
        base = os.environ.get('RPPG_EXTERNAL_COMPACT_ROOT', '')
        if not base or name != 'test' or config_data.DO_PREPROCESS:
            return False
        slug, fs, expected = DATASETS[config_data.DATASET]
        root = Path(base) / slug
        if config_data.BEGIN != 0.0 or config_data.END != 1.0:
            raise ValueError('External compact evaluation requires the complete canonical cohort')
        if config_data.FS != fs:
            raise ValueError(f'{slug}: expected native nominal sampling rate {fs}')
        pre = config_data.PREPROCESS
        crop = pre.CROP_FACE
        if not pre.DO_CHUNK or pre.RESIZE.H != pre.RESIZE.W or list(pre.DATA_AUG) != ['None']:
            raise ValueError('Unsupported external compact chunk/resize/augmentation settings')
        if pre.USE_PSUEDO_PPG_LABEL or not crop.DO_CROP_FACE or crop.BACKEND != 'HC' or not crop.USE_LARGE_FACE_BOX or crop.LARGE_BOX_COEF != 1.5 or crop.DETECTION.DO_DYNAMIC_DETECTION or crop.DETECTION.USE_MEDIAN_FACE_BOX:
            raise ValueError('External compact preprocessing differs from the pinned first-frame Haar protocol')
        metadata = json.loads((root / 'cache_manifest.json').read_text())
        rows = list(csv.DictReader((root / 'manifest.csv').open()))
        if metadata['dataset'] != slug or len(rows) != expected or len({r['session'] for r in rows}) != expected:
            raise ValueError(f'{slug}: canonical cohort mismatch')
        if sorted(r['session'] for r in rows) != sorted(metadata['sessions']):
            raise ValueError(f'{slug}: manifest session identity mismatch')
        if slug == 'ubfc_phys':
            filt = config_data.FILTERING
            if not filt.USE_EXCLUSION_LIST or set(filt.EXCLUSION_LIST) != set(metadata['excluded_sessions']) or not filt.SELECT_TASKS or set(filt.TASK_LIST) != {'T1', 'T2', 'T3'}:
                raise ValueError('UBFC-Phys must use the fixed public 101-session exclusion policy')
        if slug == 'mmpd':
            allowed = {'LIGHT': [1, 2, 3, 4], 'MOTION': [1, 2, 3, 4], 'EXERCISE': [1, 2],
                       'SKIN_COLOR': [3, 4, 5, 6], 'GENDER': [1, 2], 'GLASSER': [1, 2],
                       'HAIR_COVER': [1, 2], 'MAKEUP': [1, 2]}
            if any(set(getattr(config_data.INFO, k)) != set(v) for k, v in allowed.items()):
                raise ValueError('MMPD compact evaluation requires all 655 canonical sessions')
        self.dataset_name, self.raw_data_path, self.config_data = name, data_path, config_data
        self.data_format, self.cached_path = config_data.DATA_FORMAT, str(root)
        self.file_list_path, self.do_preprocess = str(root / 'manifest.csv'), False
        self.external_compact_cache = CompactCache(root, pre.RESIZE.H, max_sessions=8)
        self._external_clips = []
        for row in sorted(rows, key=lambda r: r['session']):
            n = int(row['frames'])
            if int(row['start_frame']) != 0 or int(row['common_eval_frames']) != min(n // c * c for c in (128, 160, 180)):
                raise ValueError('External evaluation interval mismatch')
            self._external_clips.extend((row['session'], i, i * pre.CHUNK_LENGTH, (i + 1) * pre.CHUNK_LENGTH)
                                        for i in range(n // pre.CHUNK_LENGTH))
        self.inputs = [str(root / f'{s}_input{i}.npy') for s, i, _, _ in self._external_clips]
        self.labels = [p.replace('_input', '_label') for p in self.inputs]
        self.preprocessed_data_len = len(self.inputs)
        print(f'[external compact] {slug}: {len(rows)} sessions, {len(self.inputs)} virtual clips')
        return True

    def __getitem__(self, index):
        if self.external_compact_cache is None:
            return super().__getitem__(index)
        session, chunk, start, stop = self._external_clips[index]
        pre = self.config_data.PREPROCESS
        data, label = self.external_compact_cache.clip(session, start, stop, pre.DATA_TYPE, pre.LABEL_TYPE)
        if self.data_format == 'NDCHW':
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'NCDHW':
            data = np.transpose(data, (3, 0, 1, 2))
        elif self.data_format != 'NDHWC':
            raise ValueError(f'Unsupported data format: {self.data_format}')
        return data, label, session, str(chunk)
