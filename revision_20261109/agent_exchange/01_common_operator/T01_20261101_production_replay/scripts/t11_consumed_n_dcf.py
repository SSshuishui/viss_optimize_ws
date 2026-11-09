"""T11 consumed-n correction and explicitly frozen legacy DCF extension.

Derived from archived nm1 fixture script; this diagnostic uses physical n in
both exact forward and approximate backprojection, matching production's
host-corrected consumed visibility. It is CPU subset evidence only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()

def field(rows, *names):
    for name in names:
        if name in rows.dtype.names:
            return np.atleast_1d(rows[name])
    raise ValueError('Missing field: ' + ','.join(names))

def metrics(ref, out, valid):
    x = np.asarray(ref[valid], dtype=np.float64); y = np.asarray(out[valid], dtype=np.float64)
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return {'n': len(x), 'valid_inputs': False}
    dx = x - x.mean(); dy = y - y.mean(); sx = float(dx @ dx); sy = float(dy @ dy); cov = float(dx @ dy)
    raw = float(np.sqrt(np.mean((x - y) ** 2)))
    norm_ref = float(np.linalg.norm(x))
    return {'n': len(x),
            'pearson': cov / math.sqrt(sx * sy) if sx and sy else None,
            'raw_relative_l2': float(np.linalg.norm(x - y) / norm_ref) if norm_ref else None,
            'raw_rmse': raw,
            'value_ratio_mean': float(np.mean(y[np.abs(x) > 1e-12] / x[np.abs(x) > 1e-12])) if np.any(np.abs(x) > 1e-12) else None}

def fit_plane(uvw):
    a = np.asarray(uvw, dtype=np.longdouble)
    u, v, w = a.T
    su2 = np.sum(u * u); sv2 = np.sum(v * v); suv = np.sum(u * v); suw = np.sum(u * w); svw = np.sum(v * w)
    denom = su2 * sv2 - suv * suv; eps = np.longdouble(1e-24) * (su2 * sv2 if su2 * sv2 > 0 else 1)
    if not np.isfinite(denom) or abs(denom) < eps: denom = eps if denom >= 0 else -eps
    fa = np.float32((sv2 * suw - suv * svw) / denom); fb = np.float32((su2 * svw - suv * suw) / denom)
    return float(fa), float(fb)

def grid_du(fa, fb, scale):
    lm = np.float32(max(np.sqrt(np.float32(1 + np.float32(fa) * np.float32(fa))),
                        np.sqrt(np.float32(1 + np.float32(fb) * np.float32(fb)))))
    temp = np.float32(np.ceil(np.float32(2 * np.float32(100000) / np.float32(30) * np.float32(2) * lm)))
    base = int(np.float32(temp + np.float32(3) - np.fmod(temp, np.float32(2)))) - 1
    intervals = max(2, int(math.ceil(base / scale)))
    if intervals & 1: intervals += 1
    return float(np.float32(np.float32(2) * np.float32(100000) / np.float32(30) / np.float32(intervals)))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--pixels', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--full-day-uvw', type=Path)
    p.add_argument('--legacy-dcf-bin', type=Path)
    p.add_argument('--grid-scale', type=float, default=1.0)
    p.add_argument('--npz-out', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    t_start = time.time()
    g = np.genfromtxt(args.baselines, delimiter=',', names=True, dtype=None, encoding='utf-8')
    px = np.genfromtxt(args.pixels, delimiter=',', names=True, dtype=None, encoding='utf-8')
    if 'nm1' not in px.dtype.names and 'nm1_f32' not in px.dtype.names:
        raise ValueError('pixels CSV lacks frozen nm1 column; refusing fallback to n')
    xyz = np.c_[field(px, 'l'), field(px, 'm'), field(px, 'n')].astype(np.float32).astype(np.float64)
    uvw_all = np.c_[field(g, 'u'), field(g, 'v'), field(g, 'w')].astype(np.float32)
    masks = np.fromfile(args.mask, dtype=np.uint8).reshape(len(g), len(px))
    if not np.all((masks == 0) | (masks == 1)): raise ValueError('Not a binary production mask')
    day = field(g, 'day').astype(int); take = day == 1
    t = field(g, 'global_time_index').astype(int)[take]
    if 'segment_id' in g.dtype.names: seg = field(g, 'segment_id').astype(int)[take]
    else: seg = t // 6981
    seg = seg - seg.min()
    u = uvw_all[take].astype(np.float64); mask = masks[take].astype(bool)
    weights = np.ones(len(u), dtype=np.float64)
    dcf_meta = None
    if args.legacy_dcf_bin:
        import struct
        with args.legacy_dcf_bin.open('rb') as stream:
            header = struct.unpack('<8sI8sffiQQQ', stream.read(56))
            magic,version,btag,lam,blmax,days,seed,mb_len,dcf_len = header
            if magic != b'DCFMBIN1' or version != 1: raise ValueError('bad legacy DCF header')
            stream.seek(mb_len * 8, 1)
            dcf = np.frombuffer(stream.read(dcf_len * 4), dtype='<f4').copy()
        uv = u.astype(np.float32)
        bl = np.sqrt(np.sum(uv * uv, axis=1, dtype=np.float32)).astype(np.float32)
        gs = (np.ceil((bl - np.float32(.25)) / np.float32(.5)).astype(np.int64) + 1).clip(0,len(dcf)-1)
        ss = np.clip(uv[:,2] / np.maximum(bl,np.float32(1e-20)), -1, 1)
        cc = np.sqrt(np.maximum(np.float32(0), np.float32(1)-ss*ss))
        gdg = np.divide(np.float32(1.5)*np.sqrt(np.abs(np.float32(1)-np.float32(4)*ss*ss)),cc,out=np.zeros_like(cc),where=cc>1e-12)
        weights = (np.float32(2)*np.clip(dcf[gs]*gdg,0,.125)).astype(np.float64)
        dcf_meta = {'path': str(args.legacy_dcf_bin), 'sha256': digest(args.legacy_dcf_bin),
                    'header': {'btag': btag.rstrip(b'\0').decode(), 'lambda_m': lam, 'bl_max': blmax, 'days':days, 'seed':seed},
                    'weight_min':float(weights.min()),'weight_max':float(weights.max()),'nonzero':int(np.count_nonzero(weights)),
                    'note':'frozen historical radial table x directional factor, clip[0,.125], halfset factor2; CPU float32 implementation, not claimed standard Jacobian'}
    E = np.exp(-2j * np.pi * (u @ xyz.T)) * mask          # 1120 x 2048 complex128
    valid_count = mask.sum(axis=0); valid = valid_count > 0
    vis_sky = E @ field(px, 'bp32', 'bp', 'B_area').astype(np.float32).astype(np.float64)
    dirty_direct = np.real(E.conj().T @ (weights * vis_sky))
    dirty_peak = float(np.max(np.abs(dirty_direct[valid])))
    dirty_norm = dirty_direct / dirty_peak
    # STS prepared pipeline, identical to nm1 operator-chain batch, scale fixed.
    prepared = []
    full = None
    plane_source = 'sampled exported baseline subset only'
    if args.full_day_uvw:
        full = np.fromfile(args.full_day_uvw, dtype='<f4').reshape(-1, 3)
        if len(full) != 69810 * 28: raise ValueError('Expected full day1 half-baseline fixture')
        plane_source = 'full day1 production FP32 half-uvw, contiguous ten 6981-timestamp segments'
    for sid in np.unique(seg):
        idx = np.flatnonzero(seg == sid); uu = u[idx]
        fit_data = full[int(sid) * 6981 * 28:(int(sid) + 1) * 6981 * 28] if full is not None else uu
        fa, fb = fit_plane(fit_data); du = grid_du(fa, fb, args.grid_scale)
        keys = np.floor(uu[:, :2] / du + 0.5).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        groups = [np.flatnonzero(inverse == i) for i in range(int(inverse.max()) + 1)]
        plane = 2 * np.pi * (uu[:, 0, None] * (xyz[:, 0] + fa * xyz[:, 2]) +
                             uu[:, 1, None] * (xyz[:, 1] + fb * xyz[:, 2]))
        qcoords = np.array([keys[m[0]] for m in groups]) * du
        qphase = 2 * np.pi * (qcoords[:, 0, None] * (xyz[:, 0] + fa * xyz[:, 2]) +
                              qcoords[:, 1, None] * (xyz[:, 1] + fb * xyz[:, 2]))
        prepared.append((idx, groups, np.exp(1j * plane), np.exp(1j * qphase), fa, fb, du))

    def sts_variants(y):
        out = {k: np.zeros(len(px)) for k in ['plane_raw', 'quantized_member_sum_raw', 'group_mean_raw', 'midpoint_normalized']}
        group_count = np.zeros(len(px)); mid_count = np.zeros(len(px))
        for idx, groups, eplane, eq, fa, fb, du in prepared:
            mm = mask[idx]; yy = y[idx] * weights[idx]
            out['plane_raw'] += np.real(np.sum(mm * yy[:, None] * eplane, axis=0))
            for q, members in enumerate(groups):
                occ = mm[members]; count = occ.sum(axis=0)
                sums = np.sum(occ * yy[members, None], axis=0)
                means = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
                out['quantized_member_sum_raw'] += np.real(sums * eq[q])
                out['group_mean_raw'] += np.real(means * eq[q]); group_count += count > 0
                mid = members[len(members) // 2]
                out['midpoint_normalized'] += np.real(mm[mid] * yy[mid] * eq[q]); mid_count += mm[mid]
        out['group_mean_raw'] = np.divide(out['group_mean_raw'], group_count,
                                          out=np.zeros(len(px)), where=group_count > 0)
        out['midpoint_normalized'] = np.divide(out['midpoint_normalized'], mid_count,
                                               out=np.zeros(len(px)), where=mid_count > 0)
        return out

    # ---- point-source selection (8 sources, all distinct) ----
    nm1 = xyz[:, 2] - 1.0                    # n - 1 ; n = nm1 + 1 in (0..1]
    frac = valid_count / len(u)
    order = np.argsort(nm1)
    valid_idx = np.flatnonzero(valid)
    def q_pixel(q): return int(valid_idx[int(round(q * (len(valid_idx) - 1)))])
    picks = {'zenith_nmax': int(np.argmax(nm1)),
             'n_q25': q_pixel(0.25), 'n_q50': q_pixel(0.5), 'n_q75': q_pixel(0.75),
             'horizon_nmin': int(np.argmin(nm1)),
             'vis_min': int(valid_idx[np.argmin(frac[valid_idx])]),
             'vis_mid': int(valid_idx[np.argmin(np.abs(frac - 0.5)[valid_idx])]),
             'vis_max': int(valid_idx[np.argmax(frac[valid_idx])])}
    assert len(set(picks.values())) == len(picks), 'point sources must be distinct pixels'
    variant_names = ['plane_raw', 'quantized_member_sum_raw', 'group_mean_raw', 'midpoint_normalized']
    beams = {}; beams_norm = {}; mid_beams = {}
    source_rows = []
    for name, s in picks.items():
        y = E[:, s].copy()
        beam = np.real(E.conj().T @ (weights * y))
        peak = float(beam[s])
        if peak <= 0: raise ValueError('non-positive direct peak for source ' + name)
        bnorm = beam / peak
        variants = sts_variants(y)
        rows = {}
        for vn in variant_names:
            m = metrics(beam, variants[vn], valid)
            m['value_at_source'] = float(variants[vn][s]); m['value_ratio_vs_direct_peak'] = float(variants[vn][s] / peak)
            # Shape comparison: each beam divided by its own value at the source
            # pixel, so amplitude/normalisation conventions (raw backprojection
            # vs group-normalised STS) do not dominate the metric. The raw
            # amplitude gap itself is operator evidence and is kept in
            # value_ratio_vs_direct_peak.
            vp = float(variants[vn][s])
            if vp > 0:
                m['shape_vs_direct_normalized'] = metrics(bnorm, variants[vn] / vp, valid)
            rows[vn] = m
        corr_dirty = None
        dx = bnorm[valid] - bnorm[valid].mean(); dy = dirty_norm[valid] - dirty_norm[valid].mean()
        denom = math.sqrt(float(dx @ dx) * float(dy @ dy))
        if denom: corr_dirty = float(dx @ dy / denom)
        src_mask = mask[:, s]
        rows['visible_baseline_count'] = int(src_mask.sum())
        rows['visible_fraction'] = float(src_mask.mean())
        rows['pearson_with_realsky_dirty_norm'] = corr_dirty
        beams[name] = beam; beams_norm[name] = bnorm; mid_beams[name] = variants['midpoint_normalized']
        source_rows.append({'source': name, 'pixel_index': int(s),
                            'l': float(xyz[s, 0]), 'm': float(xyz[s, 1]), 'n': float(nm1[s] + 1.0), 'nm1': float(nm1[s]),
                            'direct_peak': peak, 'variant_metrics': rows})
    # ---- shift-variance of the normalized direct beams across source pairs ----
    names = list(picks.keys())
    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            si, sj = picks[names[i]], picks[names[j]]
            both = valid.copy()
            if both.sum() < 8: continue
            bi = beams_norm[names[i]][both]; bj = beams_norm[names[j]][both]
            rel = float(np.linalg.norm(bi - bj) / max(np.linalg.norm(bi), 1e-300))
            dx = bi - bi.mean(); dy = bj - bj.mean(); den = math.sqrt(float(dx @ dx) * float(dy @ dy))
            pair_rows.append({'source_a': names[i], 'source_b': names[j],
                              'common_valid_pixels': int(both.sum()),
                              'normalized_beam_rel_l2': rel,
                              'pearson': float(dx @ dy / den) if den else None})
    rels = [r['normalized_beam_rel_l2'] for r in pair_rows]
    report = {'batch_id': 'T11-N-CONSUMED-FROZEN-LEGACY-20261101',
              'kind': 'CPU same-source direct-versus-STS and unit-versus-frozen-DCF dirty-beam diagnostics',
              'cross_source_caution': 'Different-source response maps were not recentered or rotated into a common tangent frame. Their pairwise L2 does not prove PSF shape shift-variance. Position-dependent amplitude evidence is per-source visible count and diagonal peak.',
              'phase_convention': 'n frozen fp32 column, matching actual consumed visibility after production host phase correction; exact forward and adjoint both n',
              'legacy_dcf': dcf_meta,
              'analysis_script_sha256': digest(Path(__file__)),
              'source_files': {str(path): digest(path) for path in
                               [args.baselines, args.pixels, args.mask] + ([args.full_day_uvw] if args.full_day_uvw else [])},
              'grid_scale': args.grid_scale, 'grid': [{'segment': int(sid), 'fa': it[-3], 'fb': it[-2], 'du': it[-1]}
                                                      for sid, it in zip(np.unique(seg), prepared)],
              'weights': 'unit half-baseline weights (W=I)' if dcf_meta is None else 'legacy weights on backprojection only; group denominator is original visible-group count; interpret weighted STS diagnostic explicitly',
              'realsky_dirty_peak': dirty_peak,
              'sources': source_rows,
              'unregistered_different_source_response_pairs': pair_rows,
              'unregistered_pair_difference_summary': {'n_pairs': len(rels),
                                         'normalized_beam_rel_l2_min': min(rels), 'mean': float(np.mean(rels)), 'max': max(rels)} if rels else None,
              'elapsed_setup_and_compute_s': time.time() - t_start,
              'limitations': ['Restricted fixture (1120 day1 half-baselines x 2048 exported pixels), not full-sky imaging',
                              'CPU complex128; not a bitwise CUDA STS replay; plane/quantized/midpoint follow the binary64 mathematical definitions',
                              'W=I control or frozen legacy DCF diagnostic; noiseless beams; not CUDA STS bitwise replay',
                              'Beam rel L2 values characterize operator mismatch, not image-domain sky-recovery error',
                              'Cross-source response map pairs have different source centers and are not shape shift-variance tests',
                              'Shared-node elapsed time is not a GPU/CPU performance measurement']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(args.npz_out,
                        beams=np.stack([beams[k] for k in names]),
                        beams_norm=np.stack([beams_norm[k] for k in names]),
                        mid_beams=np.stack([mid_beams[k] for k in names]),
                        dirty_direct=dirty_direct, dirty_norm=dirty_norm,
                        source_names=np.array(names), source_pixels=np.array([picks[k] for k in names]),
                        valid=valid)
    print(json.dumps({'output': str(args.output), 'sources': len(picks), 'pairs': len(pair_rows),
                      'unregistered_pair_difference_max_rel_l2': report['unregistered_pair_difference_summary']['max'] if rels else None,
                      'elapsed_s': time.time() - t_start}))

if __name__ == '__main__':
    main()
