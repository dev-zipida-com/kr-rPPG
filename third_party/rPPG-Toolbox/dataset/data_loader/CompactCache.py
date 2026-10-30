"""Lossless uint8 session crops with full-session normalization and virtual clips.

Statistics are computed before chunking. A difference clip reads its next frame,
including across a chunk boundary; only the final session sample is zero padded.
The builder owns the manifest and statistics schema; this reader never writes it.
"""
import json
import hashlib
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np
from .PackedCache import PackedFrames


class CompactCache:
    def __init__(self, root, size, max_sessions=8):
        self.root = Path(root)
        self.size = int(size)
        self.max_sessions = max_sessions
        self._sessions = OrderedDict()
        self._usage_seen = set()

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_sessions'] = OrderedDict()  # reopen mappings in each DataLoader worker
        state['_usage_seen'] = set()
        return state

    def session(self, name):
        if name in self._sessions:
            self._sessions.move_to_end(name)
            return self._sessions[name]
        folder = self.root / name
        stats = json.loads((folder / 'stats.json').read_text())
        source = folder / f'crop_{self.size}.npy'
        if os.environ.get('KR_RPPG_PACKED_CACHE', '0') == '1' and source.with_suffix('.zlib.json').exists():
            frames = PackedFrames(source, max_blocks=4)
        else:
            frames = np.load(source, mmap_mode='r', allow_pickle=False)
        bvp = np.load(folder / 'bvp.npy', mmap_mode='r', allow_pickle=False)
        if frames.dtype != np.uint8 or frames.shape != (int(stats['frames']), self.size, self.size, 3):
            raise ValueError(f'{name}: crop dtype/shape differs from compact manifest')
        if bvp.shape != (len(frames),) or bvp.dtype != np.float64:
            raise ValueError(f'{name}: BVP dtype/shape differs from compact manifest')
        self._record_usage(name, frames, source, stats)
        item = (frames, bvp, stats)
        self._sessions[name] = item
        if len(self._sessions) > self.max_sessions:
            _, old = self._sessions.popitem(last=False)
            for array in old[:2]:
                if isinstance(array, PackedFrames):
                    array.close()
                else:
                    array._mmap.close()
        return item

    def _record_usage(self, name, frames, source, stats):
        """Record each worker's first actual read, without hashing full source crops again."""
        folder = os.environ.get('RPPG_JOB_PROVENANCE_DIR', '')
        if not folder or name in self._usage_seen:
            return
        record = {'pid': os.getpid(), 'session': name, 'size': self.size,
                  'source_npy': str(source), 'frames': stats['frames']}
        if isinstance(frames, PackedFrames):
            record.update(storage='zlib', index_sha256=frames.index_sha256,
                          raw_array_sha256=frames.index['raw_array_sha256'],
                          stats_sha256=frames.index['stats_sha256'])
        else:
            stat = source.stat()
            record.update(storage='npy', source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                          stats_sha256=hashlib.sha256((source.parent / 'stats.json').read_bytes()).hexdigest())
        target = Path(folder)
        target.mkdir(parents=True, exist_ok=True)
        with (target / f'compact_usage_{os.getpid()}.jsonl').open('a') as handle:
            handle.write(json.dumps(record, sort_keys=True) + '\n')
        self._usage_seen.add(name)

    def clip(self, name, start, stop, data_types, label_type):
        frames, bvp, stats = self.session(name)
        if not 0 <= start < stop <= len(frames):
            raise ValueError(f'{name}: invalid clip interval {start}:{stop}')
        values = np.asarray(frames[start:stop], dtype=np.float32)
        scale = stats['sizes'][str(self.size)]
        transformed = []
        with np.errstate(divide='ignore', invalid='ignore'):
            for kind in data_types:
                if kind == 'Raw':
                    data = values
                elif kind == 'Standardized':
                    data = np.empty_like(values)
                    np.subtract(values, np.float32(scale['mean']), out=data)
                    np.divide(data, np.float32(scale['std']), out=data)
                elif kind == 'DiffNormalized':
                    data = np.empty_like(values)
                    valid_stop = min(stop, len(frames) - 1)
                    count = max(0, valid_stop - start)
                    if count:
                        following = np.asarray(frames[start + 1:valid_stop + 1], dtype=np.float32)
                        current = values[:count]
                        # Keep the original float32 operation order while reusing buffers.
                        np.subtract(following, current, out=data[:count])
                        np.add(following, current, out=following)
                        np.add(following, np.float32(1e-7), out=following)
                        np.divide(data[:count], following, out=data[:count])
                        np.divide(data[:count], np.float32(scale['diff_std']), out=data[:count])
                    data[count:] = 0
                else:
                    raise ValueError(f'Unsupported data type: {kind}')
                data[np.isnan(data)] = 0
                transformed.append(data)
            label = np.array(bvp[start:stop], dtype=np.float64)
            if label_type in ('DiffNormalized', 'DiffOnly'):
                label.fill(0)
                valid_stop = min(stop, len(bvp) - 1)
                count = max(0, valid_stop - start)
                if count:
                    label[:count] = bvp[start + 1:valid_stop + 1] - bvp[start:valid_stop]
                    if label_type == 'DiffNormalized':
                        label[:count] /= stats['label']['diff_std']
            elif label_type == 'Standardized':
                label = (label - stats['label']['mean']) / stats['label']['std']
            elif label_type != 'Raw':
                raise ValueError(f'Unsupported label type: {label_type}')
            label[np.isnan(label)] = 0
        data = transformed[0] if len(transformed) == 1 else np.concatenate(transformed, axis=-1)
        return data, label.astype(np.float32)
