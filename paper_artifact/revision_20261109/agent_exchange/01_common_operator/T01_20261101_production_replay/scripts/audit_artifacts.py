#!/usr/bin/env python3
"""Final read-only identity, shape, cache-stage, and storage audit."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

FIELDS=('u','v','w','x1','y1','z1','inv1','x2','y2','z2','inv2','raw_re','raw_im','re','im')
def digest(p):
    d=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024*1024),b''):d.update(b)
    return d.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();root=args.root
    freeze=json.loads((root/'freeze.json').read_text());checks=[]
    for name,expected in freeze['sha256'].items():checks.append({'kind':'frozen_source_binary_input','path':name,'pass':digest(Path(name))==expected})
    originals=json.loads((root/'production_source_identity.json').read_text())
    production=Path('/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/final_method_sensitivity_profile')
    for name,expected in originals.items():checks.append({'kind':'original_production_unchanged','path':str(production/name),'pass':digest(production/name)==expected})
    cache=[]
    dtype=np.dtype([(name,'<f4') for name in FIELDS])
    for day in (1,113,225,337,450):
        total=0;rows_for_stage=[]
        for seg in range(1,11):
            path=root/'cache'/f'day{day}_seg{seg}.bin'
            with path.open('rb') as f:header=struct.unpack('<8s7i',f.read(36))
            magic,nside,actualday,actualseg,t0,tlen,nhalf,gpus=header
            good=magic==b'PREPLAY1' and nside==4096 and actualday==day and actualseg==seg and t0==total and nhalf==28*tlen and gpus==1 and path.stat().st_size==36+60*nhalf
            checks.append({'kind':'cache_shape_identity_continuity','path':str(path),'pass':good});total+=tlen
            data=np.memmap(path,dtype=dtype,offset=36,mode='r',shape=(nhalf,))
            checks.append({'kind':'cached_all_fields_finite','path':str(path),'pass':bool(np.all(np.isfinite(data.view('<f4'))))})
            take=np.linspace(0,nhalf-1,16,dtype=int);rows_for_stage.append(np.array(data[take]));del data
            cache.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path),'header':list(header[1:])})
        sample=np.concatenate(rows_for_stage)
        # Independent host-float phase correction check on deterministic records.
        angle=np.float32(np.float32(-2)*np.float32(np.pi))*sample['w']
        c=np.cos(angle);s=np.sin(angle)
        re=sample['raw_re']*c-sample['raw_im']*s;im=sample['raw_re']*s+sample['raw_im']*c
        calculated=re.astype('f8')+1j*im.astype('f8');stored=sample['re'].astype('f8')+1j*sample['im'].astype('f8')
        error=float(np.linalg.norm(calculated-stored)/max(np.linalg.norm(stored),1e-300))
        checks.append({'kind':'raw_nm1_to_consumed_n_host_float_rotation','day':day,'samples':len(sample),'relative_l2':error,'pass':error<3e-7,
                       'definition':'cached corrected V ~= cached raw V * exp(-2pi*i*cached w), using production float angle; not a second forward simulation'})
        checks.append({'kind':'full_day_half_baselines','day':day,'count':total*28,'pass':total*28==1954680})
    completed=[]
    for path in sorted((root/'checkpoints').glob('*.json')):
        record=json.loads(path.read_text())
        good=record.get('state')=='completed' and record.get('returncode')==0
        for name,expected in record.get('artifacts',{}).items():good=good and Path(name).is_file() and digest(Path(name))==expected
        checks.append({'kind':'completed_case_artifacts','path':str(path),'pass':good});completed.append(record.get('name'))
    for case in sorted((root/'runs').glob('day*')):
        metrics=json.loads((case/'metrics.json').read_text())
        natural=np.fromfile(case/'natural_W.f64',dtype='<f8')
        pair=metrics['comparisons']['rep_vs_direct']
        good=pair['regions']['all']['n']==int((natural>0).sum())
        checks.append({'kind':'fixed_reference_valid_metric_domain','case':case.name,'reference_valid':int((natural>0).sum()),
                       'candidate_W0_retained':pair['coverage']['candidate_W0_on_reference_valid'],'pass':good})
        files=list(case.glob('*.f32'))+list(case.glob('*.f64'))
        finite=all(np.all(np.isfinite(np.fromfile(p,dtype='<f4' if p.suffix=='.f32' else '<f8'))) for p in files)
        checks.append({'kind':'all_sampled_arrays_finite_no_silent_nonfinite_exclusion','case':case.name,'pass':bool(finite)})
    paths=[p for p in root.rglob('*') if p.is_file()];total_bytes=sum(p.stat().st_size for p in paths)
    checks.append({'kind':'batch_disk_budget_under2GiB','bytes':total_bytes,'pass':total_bytes<2*1024**3})
    result={'checks':checks,'all_pass':all(c['pass'] for c in checks),'completed_cases':completed,'cache_artifacts':cache,'total_bytes':total_bytes,
        'bytes_exclude':'no filesystem-level overhead; preexisting source sky/DCF remain outside this batch','cache_stage':'both host-reduced raw nm1 and actual production host-phase-corrected consumed n; invnorms copied from GPU'}
    (root/'artifact_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'all_pass':result['all_pass'],'checks':len(checks),'completed_cases':len(completed),'batch_MiB':total_bytes/1024**2},indent=2))
    if not result['all_pass']:raise RuntimeError('artifact audit failed')
if __name__=='__main__':main()
