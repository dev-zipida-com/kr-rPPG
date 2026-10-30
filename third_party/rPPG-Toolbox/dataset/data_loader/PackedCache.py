"""Lossless indexed zlib blocks for uint8 compact crops (optional storage only)."""
import hashlib
import json
import math
import zlib
from collections import OrderedDict
from pathlib import Path
import numpy as np

BLOCK_FRAMES = 256
SCHEMA = 'compact-zlib-v1'

def source_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    with path.open('rb') as handle:
        first = handle.read(16384)
        handle.seek(max(0, stat.st_size - 16384))
        last = handle.read(16384)
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
            'selected_bytes_sha256': hashlib.sha256(first + last).hexdigest(),
            'scope': 'first and last 16KiB, size, mtime_ns; raw-array SHA computed during packing'}

class PackedFrames:
    """NumPy-like chronological slices backed by a bounded decompressed-block LRU."""
    def __init__(self, source_npy, max_blocks=4):
        self.source_npy = Path(source_npy)
        self.pack_path = self.source_npy.with_suffix('.zlib')
        self.index_path = self.source_npy.with_suffix('.zlib.json')
        index_bytes = self.index_path.read_bytes()
        self.index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        self.index = json.loads(index_bytes)
        index = self.index
        if index['schema_version'] != SCHEMA or index['block_frames'] != BLOCK_FRAMES or index['dtype'] != 'uint8':
            raise ValueError(f'Unsupported packed cache schema: {self.index_path}')
        self.shape = tuple(index['shape']); self.dtype = np.dtype(np.uint8)
        if len(self.shape) != 4 or self.shape[-1] != 3 or self.shape[1] != self.shape[2]:
            raise ValueError('Packed crop shape must be THWC square RGB')
        stats_path = self.source_npy.parent / 'stats.json'
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != index['stats_sha256']:
            raise ValueError('Packed crop statistics fingerprint changed')
        stats = json.loads(stats_path.read_text())
        if self.shape[0] != stats['frames'] or str(self.shape[1]) not in stats['sizes']:
            raise ValueError('Packed shape differs from session statistics')
        if self.source_npy.exists():
            source_stat = self.source_npy.stat()
            if (source_stat.st_size != index['source_fingerprint']['bytes'] or
                    source_stat.st_mtime_ns != index['source_fingerprint']['mtime_ns']):
                raise ValueError(f'Packed crop source metadata changed: {self.source_npy}')
        expected_count = math.ceil(self.shape[0] / BLOCK_FRAMES)
        if len(index['blocks']) != expected_count:
            raise ValueError('Packed block count differs from array length')
        offset = 0
        for i, block in enumerate(index['blocks']):
            count = min(BLOCK_FRAMES, self.shape[0] - i * BLOCK_FRAMES)
            if (block['start_frame'] != i * BLOCK_FRAMES or block['frames'] != count or
                    block['offset'] != offset or block['raw_bytes'] != count * math.prod(self.shape[1:]) or
                    block['compressed_bytes'] <= 0):
                raise ValueError('Invalid packed block offsets/lengths')
            offset += block['compressed_bytes']
        if offset != index['packed_bytes'] or self.pack_path.stat().st_size != offset:
            raise ValueError('Packed file size differs from complete index')
        self._file = self.pack_path.open('rb')
        self._blocks = OrderedDict()
        self.max_blocks = int(max_blocks)

    def __len__(self):
        return self.shape[0]

    def _block(self, block_id):
        if block_id in self._blocks:
            self._blocks.move_to_end(block_id)
            return self._blocks[block_id]
        info = self.index['blocks'][block_id]
        self._file.seek(info['offset'])
        compressed = self._file.read(info['compressed_bytes'])
        if len(compressed) != info['compressed_bytes']:
            raise ValueError('Truncated packed block')
        # zlib checks its embedded checksum while decoding each requested block.
        raw = zlib.decompress(compressed)
        if len(raw) != info['raw_bytes']:
            raise ValueError('Decoded packed block length differs from index')
        data = np.frombuffer(raw, dtype=np.uint8).reshape((info['frames'],) + self.shape[1:])
        self._blocks[block_id] = data
        if len(self._blocks) > self.max_blocks:
            self._blocks.popitem(last=False)
        return data

    def __getitem__(self, selection):
        if not isinstance(selection, slice):
            raise TypeError('Packed crops support contiguous frame slices')
        start, stop, step = selection.indices(len(self))
        if step != 1:
            raise ValueError('Packed crops require chronological stride-one slices')
        if stop <= start:
            return np.empty((0,) + self.shape[1:], dtype=np.uint8)
        pieces = []
        for i in range(start // BLOCK_FRAMES, (stop - 1) // BLOCK_FRAMES + 1):
            block = self._block(i)
            begin = max(0, start - i * BLOCK_FRAMES)
            end = min(len(block), stop - i * BLOCK_FRAMES)
            pieces.append(block[begin:end])
        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)

    def close(self):
        self._blocks.clear()
        self._file.close()
