#!/usr/bin/env python3
"""Non-vacuous full-day CPU reference, spread across the complete sampled sky."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np

FIELDS=('u','v','w','x1','y1','z1','inv1','x2','y2','z2','inv2','raw_re','raw_im','re','im')
def mask_dot(l,m,n,x,y,z,inv):
    # Simulate the two successive production f32 fused multiply-adds with
    # exact-enough f64 products/sums, then explicitly round to binary32.
    a=(l*x).astype('f4')
    a=(np.float64(m)*y.astype('f8')+a.astype('f8')).astype('f4')
    a=(np.float64(n)*z.astype('f8')+a.astype('f8')).astype('f4')
    return (a*inv).astype('f4')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    dtype=np.dtype([(f,'<f4') for f in FIELDS]);results=[]
    names=sorted(p.name for p in (root/'runs').glob('day*') if (p/'metrics.json').exists())
    R=np.float32(1737.1e3);height=np.float32(300e3)
    # Exactly use the production's exported fp32 host cosine.
    for name in names:
        case=root/'runs'/name;day=int(name.split('_')[0][3:]);rotation=.37 if '_rot037_' in name else 0
        xyz=[np.fromfile(case/(a+'.f32'),dtype='<f4') for a in ('l','m','n')];npix=len(xyz[0]);ix=((np.arange(16)+.5)*npix/16).astype(int)
        L,M,N=[x[ix] for x in xyz];cosphi=np.float32(np.loadtxt(case/'oracle_config.txt')[2])
        direct=np.zeros(16);plane=np.zeros(16);W=np.zeros(16,dtype=np.int64);ambiguity=np.zeros(16)
        meta=list(csv.DictReader((case/'segments.csv').open()))
        allrows=np.concatenate([np.fromfile(root/'cache'/f'day{day}_seg{sid}.bin',dtype=dtype,offset=36) for sid in range(1,11)])
        if rotation:
            c,s=math.cos(rotation),math.sin(rotation)
            for a,b in [('u','w'),('x1','z1'),('x2','z2')]:
                aa=allrows[a].astype('f8');bb=allrows[b].astype('f8');allrows[a]=(c*aa+s*bb).astype('f4');allrows[b]=(-s*aa+c*bb).astype('f4')
        # Repartition the frozen full day using each actual segment plan.
        for segment in meta:
            t0,tlen=int(segment['t0']),int(segment['tlen']);rows=allrows[t0*28:(t0+tlen)*28]
            u,v,w=[rows[a].astype('f8') for a in ('u','v','w')];vis=rows['re'].astype('f8')+1j*rows['im'].astype('f8')
            fa=float(np.float32(segment['fa']));fb=float(np.float32(segment['fb']))
            for j in range(16):
                ll,mm,nn=L[j],M[j],N[j]
                c1=mask_dot(ll,mm,nn,rows['x1'],rows['y1'],rows['z1'],rows['inv1'])
                c2=mask_dot(ll,mm,nn,rows['x2'],rows['y2'],rows['z2'],rows['inv2'])
                mask=(c1>=cosphi)&(c2>=cosphi)
                pd=2*np.pi*(u*float(ll)+v*float(mm)+w*float(nn))
                pp=2*np.pi*(u*(float(ll)+fa*float(nn))+v*(float(mm)+fb*float(nn)))
                direct[j]+=np.real(vis[mask]*np.exp(1j*pd[mask])).sum()
                plane[j]+=np.real(vis[mask]*np.exp(1j*pp[mask])).sum();W[j]+=mask.sum()
                boundary=(np.abs(c1-cosphi)<5e-7)|(np.abs(c2-cosphi)<5e-7)
                ambiguity[j]+=np.abs(vis[boundary]).sum()
        gpuN=np.fromfile(case/'direct_N.f64',dtype='<f8')[ix];gpuP=np.fromfile(case/'plane_N.f64',dtype='<f8')[ix];gpuW=np.fromfile(case/'natural_W.f64',dtype='<f8')[ix]
        eN=float(np.linalg.norm(direct-gpuN)/max(np.linalg.norm(direct),1e-300));eP=float(np.linalg.norm(plane-gpuP)/max(np.linalg.norm(plane),1e-300))
        old=np.fromfile(case/'oracle_first_partition.f64',dtype='<f8').reshape(9,16)
        passed=eN<1e-9 and eP<1e-9 and np.array_equal(W,gpuW) and np.all(W>0)
        results.append({'case':name,'probe_sample_indices':ix.tolist(),'full_half_baselines':1954680,'directions':16,'nonzero_full_day_probes':int((W>0).sum()),
            'direct_relative_l2':eN,'plane_relative_l2':eP,'visible_count_exact':bool(np.array_equal(W,gpuW)),
            'maximum_visible_count_difference':int(np.max(np.abs(W-gpuW))),
            'old_first_partition_visible_pair_sum':float(old[7].sum()),'old_first_partition_nonzero_directions':int((old[7]>0).sum()),
            'old_local_oracle_vacuous':bool(old[7].sum()==0),'pass':bool(passed),
            'FP32_mask_boundary_absolute_response_bound_per_probe':ambiguity.tolist()})
        (case/'full_cpu_probe.json').write_text(json.dumps(results[-1],indent=2)+'\n')
        metrics=json.loads((case/'metrics.json').read_text())
        metrics['oracle']['original_local_nonvacuous']=bool(old[7].sum()>0)
        metrics['oracle']['original_local_visible_pair_sum']=float(old[7].sum())
        metrics['oracle']['original_local_criteria_pass']=metrics['oracle'].get('original_local_criteria_pass',metrics['oracle']['pass'])
        metrics['oracle']['authoritative_gate']='independent_full_day_CPU_reference'
        metrics['oracle']['pass']=bool(passed)
        metrics['independent_full_day_CPU_reference']=results[-1]
        (case/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
        print(json.dumps({k:results[-1][k] for k in ('case','direct_relative_l2','plane_relative_l2','visible_count_exact','old_local_oracle_vacuous','pass')}),flush=True)
    result={'all_pass':all(r['pass'] for r in results),'cases':results,
        'criteria':'each case16 full-sky spread directions, all1954680 half-baselines, all probes nonzero; CPU FP64 direct and plane relativeL2<1e-9 and fp32 visible counts exact',
        'correction':'the earlier first16-direction first8192-baseline local test is vacuous where all probes are occulted; this independent full-day reference replaces that vacuous portion of the numeric gate. Main imaging data and metrics are unchanged.'}
    (root/'full_cpu_oracle.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['all_pass']:raise RuntimeError('non-vacuous full CPU oracle failed')
if __name__=='__main__':main()
