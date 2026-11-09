#!/usr/bin/env python3
"""Offline, explicit-operator metrics and independent CPU control verification."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(root,name,dtype):return np.fromfile(root/name,dtype=dtype)
def finite(value):return float(value) if np.isfinite(value) else None
def compare(ref,cand,valid,z):
    valid=valid & np.isfinite(ref) & np.isfinite(cand)
    if not np.any(valid):return {'n':0}
    r,c=ref[valid],cand[valid]
    alpha,beta=np.linalg.lstsq(np.column_stack((c,np.ones_like(c))),r,rcond=None)[0]
    aligned=alpha*cand+beta
    result={'n':int(valid.sum()),'excluded':int(len(valid)-valid.sum()),'global_fit_ref_equals_alpha_candidate_plus_beta':{'alpha':float(alpha),'beta':float(beta)},'regions':{}}
    for name,region in [('all',np.ones(len(z),bool)),('low_lat_abs_z_le_0p5',np.abs(z)<=.5),('mid_lat', (np.abs(z)>.5)&(np.abs(z)<=np.sqrt(3)/2)),('high_lat',np.abs(z)>np.sqrt(3)/2),('north',z>=0),('south',z<0)]:
        keep=valid & region;r=ref[keep];c=cand[keep];a=aligned[keep]
        if len(r)<2:result['regions'][name]={'n':int(len(r))};continue
        error=c-r;std=np.std(r)
        result['regions'][name]={'n':int(len(r)),'raw_relative_l2':finite(np.linalg.norm(error)/max(np.linalg.norm(r),1e-300)),
            'raw_rmse_over_ref_std':finite(np.sqrt(np.mean(error**2))/max(std,1e-300)),
            'global_affine_rmse_over_ref_std':finite(np.sqrt(np.mean((a-r)**2))/max(std,1e-300)),
            'pearson':finite(np.corrcoef(r,c)[0,1]),'ref_mean':float(np.mean(r)),'candidate_mean':float(np.mean(c))}
    return result

def verify_oracle(root):
    dtype=np.dtype([(n,'<f4') for n in ('u','v','w','x1','y1','z1','inv1','x2','y2','z2','inv2','raw_re','raw_im','re','im')])
    r=np.fromfile(root/'oracle_rows.bin',dtype=dtype);wt=read(root,'oracle_weights.f32','<f4').astype('f8')
    gu=read(root,'oracle_gu.f32','<f4').astype('f8');gv=read(root,'oracle_gv.f32','<f4').astype('f8')
    l=read(root,'l.f32','<f4')[:16];m=read(root,'m.f32','<f4')[:16];n=read(root,'n.f32','<f4')[:16]
    fa,fb,cosphi=np.loadtxt(root/'oracle_config.txt');fa=float(fa);fb=float(fb)
    gpu=read(root,'oracle_first_partition.f64','<f8').reshape(9,16)
    cpu=np.zeros_like(gpu)
    for p in range(16):
        # Same fp32 point-mask expression; oracle phases and reductions are FP64.
        c1=(l[p]*r['x1']+m[p]*r['y1']+n[p]*r['z1'])*r['inv1']
        c2=(l[p]*r['x2']+m[p]*r['y2']+n[p]*r['z2'])*r['inv2']
        mask=(c1>=cosphi)&(c2>=cosphi)
        u=r['u'].astype('f8');v=r['v'].astype('f8');w=r['w'].astype('f8')
        lp=float(l[p])+fa*float(n[p]);mp=float(m[p])+fb*float(n[p])
        vcomplex=r['re'].astype('f8')+1j*r['im'].astype('f8')
        raw=r['raw_re'].astype('f8')+1j*r['raw_im'].astype('f8')
        phases=[2*np.pi*(u*float(l[p])+v*float(m[p])+w*float(n[p])),
                2*np.pi*(u*lp+v*mp),2*np.pi*(gu*lp+gv*mp),
                2*np.pi*(u*float(l[p])+v*float(m[p])+w*float(np.float32(n[p]-np.float32(1))))]
        values=[np.real((raw if k==3 else vcomplex)*np.exp(1j*phase)) for k,phase in enumerate(phases)]
        for k in range(4):cpu[k,p]=values[k][mask].sum()
        for k in range(3):cpu[k+4,p]=(wt*values[k])[mask].sum()
        cpu[7,p]=mask.sum();cpu[8,p]=wt[mask].sum()
    error=np.linalg.norm(cpu-gpu)/max(np.linalg.norm(cpu),1e-300)
    per_channel=[finite(np.linalg.norm(cpu[k]-gpu[k])/max(np.linalg.norm(cpu[k]),1e-300)) for k in range(9)]
    return {'relative_l2':float(error),'per_channel_relative_l2':per_channel,'weight_count_exact':bool(np.array_equal(cpu[7],gpu[7])),
            'pass':bool(error<1e-10 and all(value is not None and value<1e-9 for value in per_channel) and np.array_equal(cpu[7],gpu[7])),
            'criteria':'global relativeL2<1e-10, each of 9 channels relativeL2<1e-9, natural visible counts exactly equal',
            'scope':'first segment first8192 half-baselines x16 production directions; CPU mask expressions may differ by FMA rounding'}

def analyze(root,cache):
    z=read(root,'n_original.f32','<f4');W=read(root,'natural_W.f64','<f8');CW=read(root,'cell_W.f64','<f8');LW=read(root,'legacy_dcf_W.f64','<f8')
    maps={};weights={}
    for label,w in [('direct',W),('plane',W),('quant_member',W),('raw_nm1',W),('legacy_dcf_direct',LW),('legacy_dcf_plane',LW),('legacy_dcf_quant',LW),('cell_direct',CW),('cell_plane',CW),('cell_quant',CW)]:
        numerator=read(root,label+'_N.f64','<f8');maps[label]=np.divide(numerator,w,out=np.zeros_like(numerator),where=w>0);weights[label]=w
    for label in ('groupmean','rep','centroid'):
        numerator=read(root,label+'_N.f32','<f4').astype('f8');w=read(root,label+'_W.u32','<u4').astype('f8')
        maps[label]=np.divide(numerator,w,out=np.zeros_like(numerator),where=w>0);weights[label]=w
    pairs=[('direct','plane'),('direct','quant_member'),('direct','raw_nm1'),('direct','legacy_dcf_direct'),('direct','cell_direct'),('direct','rep'),('direct','centroid'),
           ('cell_direct','cell_plane'),('cell_direct','cell_quant'),('cell_direct','groupmean'),('cell_direct','rep'),('cell_direct','centroid'),
           ('cell_quant','groupmean'),('groupmean','rep'),('groupmean','centroid'),('rep','centroid')]
    result={'root':str(root),'samples':len(z),'operator_notes':{
        'visibility':'full201M production FP32 G1 forward; raw nm1 and consumed float host-corrected n cached separately; every imaging method consumes same corrected V',
        'natural':'per-baseline unit weight, original point mask, final sum numerator / visible baseline count',
        'cell':'exact per-pixel visible-member average inside each production cell, then sum across visible cells / visible-cell count; identity differs from natural',
        'groupmean':'unmodified production occ_mode0 kernel with cell quantization',
        'rep':'unmodified production representative gather and reconstruction kernels, baseline-sample-order midpoint',
        'centroid':'same production representative kernels, only selection changed to actual-member uv centroid-nearest, lowest half-index breaks ties',
        'legacy_dcf':'existing immutable radial-table + directional factor + cap and halfset factor; separate diagnostic changing W; no claim this is a standard Jacobian',
        'timings':'shared-node progress only; cached replay timing excludes full forward and setup from other phases'},'comparisons':{}}
    noise_meta=cache.parent/'noise_metadata.json'
    if noise_meta.exists():
        result['noise']=json.loads(noise_meta.read_text())
        result['operator_notes']['visibility']='full201M production FP32 G1 fullsky consumed-n visibility plus one recorded independent20dB complex Gaussian half-set realization; not a new forward simulation'
        result['operator_notes']['raw_nm1']='reconstructed by inverse float host rotation from noisy consumed-n values, solely a noise diagnostic'
    for a,b in pairs:
        # Freeze the reference-valid domain. A candidate's W=0 is its zero
        # output, counted as error rather than silently dropping that direction.
        item=compare(maps[a],maps[b],weights[a]>0,z)
        item['coverage']={'metric_domain':'reference W>0; candidate W0 retained as zero output',
            'reference_valid':int((weights[a]>0).sum()),'candidate_valid':int((weights[b]>0).sum()),
            'candidate_W0_on_reference_valid':int(((weights[a]>0)&(weights[b]<=0)).sum()),
            'candidate_nonzero_on_reference_invalid':int(((weights[a]<=0)&(weights[b]>0)).sum()),
            'coverage_disagreement':int(np.count_nonzero((weights[a]>0)!=(weights[b]>0)))}
        result['comparisons'][b+'_vs_'+a]=item
    result['oracle']=verify_oracle(root)
    result['production_weight_consistency']={'groupmean_vs_cell_weight_exact':bool(np.array_equal(weights['groupmean'],CW)),
        'rep_vs_groupmean_pixels_differ':int(np.count_nonzero(weights['rep']!=weights['groupmean'])),
        'centroid_vs_groupmean_pixels_differ':int(np.count_nonzero(weights['centroid']!=weights['groupmean']))}
    ids=read(root,'ids.u32','<u4')
    result['sample_coverage']={'HEALPix_NEST_face_counts':np.bincount(ids//(4096**2),minlength=12).tolist(),
        'min_z':float(z.min()),'max_z':float(z.max()),'low_lat_count':int((np.abs(z)<=.5).sum()),
        'high_lat_count':int((np.abs(z)>np.sqrt(3)/2).sum())}
    segment_rows=list(csv.DictReader((root/'segments.csv').open()))
    phase=[]
    ll=read(root,'l.f32','<f4').astype('f8');mm=read(root,'m.f32','<f4').astype('f8');nn=read(root,'n.f32','<f4').astype('f8')
    for row in segment_rows:
        fa,fb=float(row['fa']),float(row['fb']);du=float(row['du'])
        phase.append({'segment':int(row['seg']),'fit_rms_lambda':float(row['fit_rms']),'fit_max_lambda':float(row['fit_max']),
            'plane_phase_max_bound_rad_on_sample':float(2*np.pi*float(row['fit_max'])*np.abs(nn).max()),
            'plane_phase_rms_unmasked_cartesian_pairs_rad':float(2*np.pi*float(row['fit_rms'])*np.sqrt(np.mean(nn**2))),
            'quantization_phase_max_bound_rad_on_sample':float(np.max(np.pi*du*(np.abs(ll+fa*nn)+np.abs(mm+fb*nn))))})
    result['phase_bounds']={'per_segment':phase,'scope':'plane rms factorizes over baseline x sampled direction Cartesian product, without point-mask restriction; maxima are conservative bounds, not observed error maxima. Quantization bound uses per-axis rounding displacement du/2.'}
    sky=Path('/data/zhaox/research/viss_optimize_ws/earth_10Mhz/B_10M.bin')
    if sky.exists():
        B=np.asarray(np.memmap(sky,dtype='<f4',mode='r')[ids],dtype='f8')
        result['dirty_response_vs_input_sky']={'warning':'dirty-map response includes incomplete sampling/occultation and weighting; these correlations are not sky-recovery accuracy',
            'metric_domain':'fixed natural reference W>0; candidate W0 is zero output',
            'maps':{name:compare(B,maps[name],W>0,z) for name in ('direct','cell_direct','rep','centroid')}}
    arrays={'pixel_ids':ids,'l':read(root,'l.f32','<f4'),'m':read(root,'m.f32','<f4'),
        'n':read(root,'n.f32','<f4'),'n_original':z,'direct_W':W,'cell_W':CW,'legacy_dcf_W':LW}
    for label in ('direct','plane','quant_member','raw_nm1','legacy_dcf_direct','legacy_dcf_plane','legacy_dcf_quant','cell_direct','cell_plane','cell_quant'):
        arrays[label+'_N']=read(root,label+'_N.f64','<f8')
    for label in ('groupmean','rep','centroid'):
        arrays[label+'_N']=read(root,label+'_N.f32','<f4');arrays[label+'_W']=read(root,label+'_W.u32','<u4')
    if sky.exists():arrays['input_sky_B']=B.astype('<f4')
    tmp=root/'case.npz.tmp'
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    tmp.replace(root/'case.npz')
    gate=cache/'production_gate_day1_ids.u32'
    if root.name=='day1_s10_g1_n32768' and gate.exists():
        ids=read(root,'ids.u32','<u4');gate_ids=np.fromfile(gate,dtype='<u4');gateN=read(cache,'production_gate_day1_N.f32','<f4');gateW=read(cache,'production_gate_day1_W.u32','<u4')
        N=read(root,'rep_N.f32','<f4');ww=read(root,'rep_W.u32','<u4')
        result['production_full_sky_sample_gate']={'ids_bitwise':bool(np.array_equal(ids,gate_ids)),
            'N_bitwise':bool(np.array_equal(N,gateN)),'W_bitwise':bool(np.array_equal(ww,gateW)),
            'N_relative_l2':float(np.linalg.norm(N.astype('f8')-gateN)/max(np.linalg.norm(gateN.astype('f8')),1e-300))}
    result['files']={p.name:{'sha256':sha(p),'size':p.stat().st_size} for p in root.iterdir() if p.is_file() and p.name!='metrics.json'}
    (root/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);p.add_argument('--case')
    args=p.parse_args();runs=args.root/'runs'
    paths=[runs/args.case] if args.case else [path for path in sorted(runs.iterdir()) if (path/'direct_N.f64').exists()]
    out=[]
    for path in paths:
        result=analyze(path,args.root/'cache');out.append({'case':path.name,'oracle_pass':result['oracle']['pass'],'rep_vs_direct':result['comparisons']['rep_vs_direct']['regions']['all'],'rep_vs_cell_direct':result['comparisons']['rep_vs_cell_direct']['regions']['all'],'production_gate':result.get('production_full_sky_sample_gate')})
        if not result['oracle']['pass']:raise RuntimeError('independent CPU oracle failed '+path.name)
        if result.get('production_full_sky_sample_gate') and not all(result['production_full_sky_sample_gate'][k] for k in ('ids_bitwise','N_bitwise','W_bitwise')):raise RuntimeError('production sampled gate failed '+path.name)
    (args.root/'analysis_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
