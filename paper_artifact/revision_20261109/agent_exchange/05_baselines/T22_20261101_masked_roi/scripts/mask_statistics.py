#!/usr/bin/env python3
"""CPU-only exact mask-pattern census for all 36 predeclared compact ROIs."""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def nest_coordinates(ids, nside=4096):
    """Production-style NEST frame, explicit FP32 NumPy math, not GPU bitwise."""
    ids = np.asarray(ids, dtype=np.uint32)
    face = ids // (nside * nside)
    ipf = ids % (nside * nside)
    ix = np.zeros(len(ids), dtype=np.int32)
    iy = np.zeros(len(ids), dtype=np.int32)
    for bit in range(12):
        ix |= (((ipf >> (2 * bit)) & 1) << bit).astype(np.int32)
        iy |= (((ipf >> (2 * bit + 1)) & 1) << bit).astype(np.int32)
    jrll = np.asarray([2,2,2,2,3,3,3,3,4,4,4,4], dtype=np.int32)
    jpll = np.asarray([1,3,5,7,0,2,4,6,1,3,5,7], dtype=np.int32)
    jr = jrll[face] * nside - (ix + iy) - 1
    nr = np.where(jr < nside, jr, np.where(jr <= 3*nside, nside, 4*nside-jr)).astype(np.int32)
    kshift = np.where((jr >= nside) & (jr <= 3*nside), (jr-nside) & 1, 0).astype(np.int32)
    fact1 = np.float32(1.0 / (3 * nside * nside))
    fact2 = np.float32(2.0 / (3 * nside))
    z = np.where(jr < nside, np.float32(1) - (nr*nr).astype(np.float32)*fact1,
        np.where(jr <= 3*nside, (2*nside-jr).astype(np.float32)*fact2,
                 np.float32(-1) + (nr*nr).astype(np.float32)*fact1)).astype(np.float32)
    z = np.clip(z, np.float32(-1), np.float32(1))
    theta = np.arccos(z)
    jp = (jpll[face]*nr + (ix-iy) + 1 + kshift) >> 1
    jp = np.where(jp > 4*nside, jp-4*nside, jp)
    jp = np.where(jp < 1, jp+4*nside, jp)
    pi = np.float32(np.pi)
    phi = (np.float32(.5)*pi) * (jp.astype(np.float32) - np.float32(.5)*(kshift+1).astype(np.float32)) / nr.astype(np.float32)
    phi = np.where(phi < 0, phi + np.float32(2)*pi, phi).astype(np.float32)
    th = pi*np.float32(.5)-theta
    phi = -np.where(phi > pi, phi-np.float32(2)*pi, phi).astype(np.float32)
    ct = np.cos(th)
    return np.column_stack((ct*np.cos(phi), ct*np.sin(phi), np.sin(th))).astype(np.float32)


def mask_rows(lmn, endpoints1, endpoints2, invnorm, cosphi, block=32):
    """Explicit FP32 non-FMA left-associated point predicate; no cone shortcut."""
    nrows = len(endpoints1)
    out = np.empty((nrows, (len(lmn)+7)//8), dtype=np.uint8)
    l, m, n = lmn.T
    for lo in range(0, nrows, block):
        hi = min(lo+block, nrows)
        keep = np.ones((hi-lo, len(lmn)), dtype=bool)
        for e, norm in ((endpoints1, invnorm[:, 0]), (endpoints2, invnorm[:, 1])):
            dot = l[None, :] * e[lo:hi, 0, None]
            dot = dot + m[None, :] * e[lo:hi, 1, None]
            dot = dot + n[None, :] * e[lo:hi, 2, None]
            dot = dot * norm[lo:hi, None]
            keep &= dot >= cosphi
        out[lo:hi] = np.packbits(keep, axis=1, bitorder='little')
    return out


def census(lmn, rows, cosphi):
    packed = mask_rows(lmn, rows[:, 3:6], rows[:, 7:10], rows[:, [6, 10]], cosphi)
    unique, inverse, counts = np.unique(packed, axis=0, return_inverse=True, return_counts=True)
    assert np.array_equal(unique[inverse], packed)
    decoded = np.unpackbits(unique, axis=1, count=len(lmn), bitorder='little')
    visible = decoded.sum(axis=1, dtype=np.uint64)
    denom = (2*np.asarray(counts, dtype=np.uint64) @ decoded.astype(np.uint64)).astype(np.uint32)
    naive_d = 2*np.unpackbits(packed, axis=1, count=len(lmn), bitorder='little').sum(axis=0, dtype=np.uint32)
    assert np.array_equal(denom, naive_d)
    hidden = visible == 0
    full = visible == len(lmn)
    boundary = (~hidden) & (~full)
    patterns = [hashlib.sha256(p.tobytes()).hexdigest() for p in unique]
    info = {
        'baseline_rows': len(rows), 'targets': len(lmn), 'exact_mask_groups': len(unique),
        'nonzero_groups_requiring_transform': int(np.count_nonzero(~hidden)),
        'all_hidden_groups': int(hidden.sum()), 'all_visible_groups': int(full.sum()),
        'boundary_groups': int(boundary.sum()),
        'all_hidden_baseline_rows': int(counts[hidden].sum()),
        'all_visible_baseline_rows': int(counts[full].sum()),
        'boundary_baseline_rows': int(counts[boundary].sum()),
        'visible_pair_fraction': float((counts.astype(np.uint64)*visible).sum()/(len(rows)*len(lmn))),
        'baseline_rows_per_nonzero_group': counts[~hidden].tolist(),
        'visible_targets_per_nonzero_group': visible[~hidden].tolist(),
        'full_target_NUFFT_group_output_count': int(np.count_nonzero(~hidden)*len(lmn)),
        'visible_target_NUFFT_group_output_count': int(visible[~hidden].sum()),
        'D_zero_directions': int(np.count_nonzero(denom == 0)),
        'D_min_max': [int(denom.min()), int(denom.max())],
        'packed_mask_sha256': hashlib.sha256(packed.tobytes()).hexdigest(),
        'D_sha256': hashlib.sha256(denom.tobytes()).hexdigest(),
        'group_reconstruction_bitwise': True,
        'grouped_D_exact': True,
        'dense_mask_persisted': False,
    }
    arrays = {'group_id_per_baseline': inverse.astype(np.uint32), 'group_sizes': counts.astype(np.uint32),
              'group_visible_targets': visible.astype(np.uint32), 'pattern_sha256': np.asarray(patterns), 'D': denom}
    return info, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    own = Path(__file__).resolve().parents[1]
    p.add_argument('--root', type=Path, default=own)
    p.add_argument('--geometry', type=Path, default=own.parents[2] / 'experiments/revision_server_20261026/geometry/production_fp32_geometry_export_v1.npz')
    args = p.parse_args()
    root = args.root.resolve()
    # Workspace root for standalone invocation; explicit --geometry is also supported.
    if not args.geometry.exists():
        raise FileNotFoundError('Supply the frozen geometry file with --geometry')
    protocol = json.loads((root/'PROTOCOL.json').read_text())
    assert protocol['resources']['new_GPU_execution_currently_allowed'] is False
    assert protocol['ROI_selection_before_statistics']['case_count'] == 36
    started = time.time()
    inputs = np.load(root/'inputs_existing.npz', allow_pickle=False)
    rows = inputs['rows']
    cosphi = np.float32(inputs['cosphi'])
    metadata = json.loads(str(inputs['source_manifest_json']))
    geometry = np.load(args.geometry, allow_pickle=False)
    gp = geometry['lmn_fp32']
    converted = nest_coordinates(geometry['pixel_ids'])
    gmask = mask_rows(gp, geometry['endpoint1_km_fp32'], geometry['endpoint2_km_fp32'], geometry['invnorm_per_km_fp32'], cosphi)
    expected = np.packbits(geometry['point_mask_u8'], axis=1, bitorder='little')
    diff = np.unpackbits(gmask ^ expected, axis=1, count=len(gp), bitorder='little')
    validation = {
        'existing_geometry_file_sha256': sha(args.geometry),
        'CPU_NEST_vs_existing_GPU_2048_max_abs_coordinate': float(np.max(np.abs(converted-gp))),
        'CPU_NEST_vs_existing_GPU_2048_coordinate_relativeL2': float(np.linalg.norm(converted.astype('f8')-gp)/np.linalg.norm(gp.astype('f8'))),
        'specified_nonFMA_FP32_vs_existing_export_mask_disagreements': int(diff.sum()),
        'mask_comparison_pairs': int(np.prod(geometry['point_mask_u8'].shape)),
        'interpretation': 'Both benchmark implementations use the newly specified exact same frozen masks. This diagnostic does not assert GPU fast-trig or FMA predicate bitwise equivalence.',
        'source_cache_metadata': metadata,
    }
    (root/'input_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    directory = root/'cases'
    directory.mkdir(exist_ok=True)
    output = []
    for face in protocol['ROI_selection_before_statistics']['faces']:
        ids = np.arange(face*4096**2, face*4096**2+65536, dtype=np.uint32)
        coords = inputs['prefix_coords'].copy() if face == 0 else nest_coordinates(ids)
        coordinate_source = 'existing_production_GPU_prefix' if face == 0 else 'frozen_CPU_production_style_NEST'
        np.savez_compressed(directory/f'face{face:02d}_coordinates.npz', pixel_ids=ids, lmn=coords)
        for size in protocol['ROI_selection_before_statistics']['sizes']:
            label = f'face{face:02d}_n{size}'
            t0 = time.perf_counter()
            info, arrays = census(coords[:size], rows, cosphi)
            info.update(case=label, face=face, start_pixel=int(ids[0]), coordinate_source=coordinate_source,
                        coordinate_prefix_sha256=hashlib.sha256(coords[:size].tobytes()).hexdigest(),
                        CPU_mask_grouping_wall_s=time.perf_counter()-t0)
            np.savez_compressed(directory/(label+'.npz'), **arrays)
            (directory/(label+'.json')).write_text(json.dumps(info, indent=2)+'\n')
            output.append(info)
            retained = sum(q.stat().st_size for q in root.rglob('*') if q.is_file())
            assert retained <= protocol['resources']['additional_retained_bytes_max'], 'Retention budget exceeded'
            print(json.dumps({key: info[key] for key in ['case','exact_mask_groups','boundary_groups','all_visible_baseline_rows','all_hidden_baseline_rows','D_zero_directions']}), flush=True)
    result = {
        'status': 'completed_CPU_mask_statistics_GPU_not_started',
        'protocol_sha256': sha(root/'PROTOCOL.json'), 'inputs_sha256': sha(root/'inputs_existing.npz'),
        'script_sha256': sha(Path(__file__).resolve()),
        'case_count': len(output), 'cases': output, 'input_validation': validation,
        'elapsed_s': time.time()-started,
        'process_lifetime_peak_RSS_native_units': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'no_new_forward_or_GPU': True,
        'retained_bytes_before_summary': sum(q.stat().st_size for q in root.rglob('*') if q.is_file()),
    }
    (root/'mask_statistics.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
