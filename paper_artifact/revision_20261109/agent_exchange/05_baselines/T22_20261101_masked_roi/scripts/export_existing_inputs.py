#!/usr/bin/env python3
"""Read existing server files, write a small NPZ to stdout; never use CUDA."""
import hashlib
import io
import json
import struct
import sys
from pathlib import Path
import numpy as np

if len(sys.argv) != 2:
    raise SystemExit('Usage: python3 export_existing_inputs.py INPUT_DATA_ROOT')
root = Path(sys.argv[1]).resolve()
geometry = root / 'geometry/production_fp32_geometry_export_v1.npz'
expected = 'bfaeef3ee815871fd1216fff6b9c889932b2d51f5c05afccde39cf6f44f78d6b'
assert hashlib.sha256(geometry.read_bytes()).hexdigest() == expected
g = np.load(geometry, allow_pickle=False)
selected = np.flatnonzero(g['day'] == 1)
assert len(selected) == 1120
times = g['global_time_index'][selected]
pairs = [(m, n) for m in range(8) for n in range(m + 1, 8)]
pair_ids = np.asarray([pairs.index((int(g['pair_m'][j]), int(g['pair_n'][j]))) for j in selected], dtype=np.int32)
cache = root / 'agent_exchange/01_common_operator/T01_20261101_production_replay/cache'
rows = np.empty((len(selected), 15), dtype='<f4')
filled = np.zeros(len(selected), dtype=np.uint8)
metadata = []
for seg in range(1, 11):
    path = cache / f'day1_seg{seg}.bin'
    with path.open('rb') as stream:
        header = struct.unpack('<8s7i', stream.read(36))
    magic, nside, day, actual_seg, t0, tlen, nhalf, ngpu = header
    assert magic == b'PREPLAY1' and nside == 4096 and day == 1 and actual_seg == seg and ngpu == 1
    assert nhalf == tlen * 28 and path.stat().st_size == 36 + nhalf * 60
    keep = np.flatnonzero((times >= t0) & (times < t0 + tlen))
    index = (times[keep] - t0) * 28 + pair_ids[keep]
    data = np.memmap(path, dtype='<f4', mode='r', offset=36, shape=(nhalf, 15))
    rows[keep] = data[index]
    filled[keep] += 1
    metadata.append({'path': str(path), 'bytes': path.stat().st_size, 'header': [nside, day, actual_seg, t0, tlen, nhalf, ngpu],
                     'selected_rows': len(keep), 'identity_scope': 'Existing frozen cache; selected rows hashed. No fresh full-file hash in this read-only extraction.'})
    del data
assert np.all(filled == 1)
uv_error = rows[:, :3] - g['uvw_lambda_fp32'][selected]
assert np.max(np.abs(uv_error)) == 0, 'Cache mapping is not identical to frozen exported UVW'
prefix = root / 'precision/queued_correctness_v2/pilot8_dump_v2/prefix_dump'
coords = np.column_stack([np.fromfile(prefix / name, dtype='<f4', count=65536) for name in ['l.f32', 'm.f32', 'n.f32']])
assert coords.shape == (65536, 3)
manifest = {'geometry_path': str(geometry), 'geometry_sha256': expected, 'cache_inputs': metadata,
            'UVW_export_mapping_bitwise': True, 'selected_rows_sha256': hashlib.sha256(rows.tobytes()).hexdigest(),
            'prefix_coords_sha256': hashlib.sha256(coords.tobytes()).hexdigest(), 'no_forward_or_GPU': True,
            'record_fields': ['u','v','w','x1','y1','z1','inv1','x2','y2','z2','inv2','raw_re','raw_im','re','im']}
buffer = io.BytesIO()
np.savez_compressed(buffer, rows=rows, source_geometry_indices=selected.astype('<i4'), global_time_index=times.astype('<i4'),
                    pair_id=pair_ids, prefix_coords=coords, cosphi=np.asarray(g['cosphi_fp32']),
                    source_manifest_json=np.asarray(json.dumps(manifest)))
sys.stdout.buffer.write(buffer.getvalue())
