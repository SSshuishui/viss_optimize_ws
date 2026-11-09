#!/usr/bin/env python3
"""Small cached-V numerical/race gate. No forward, full map or timing claim."""
import argparse,csv,datetime,fcntl,hashlib,json,os,signal,subprocess,sys
from pathlib import Path
import numpy as np

TARGET_UUID='GPU-99fbd651-f080-aa22-391b-f33caa723578'
LIMIT_BYTES=100*1024**2
MIN_FREE=100*1024**3

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def compare(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');d=a-b
    return {'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-300)),
            'max_abs_over_peak':float(np.max(np.abs(d))/max(np.max(np.abs(b)),1e-300)),
            'pearson':float(np.corrcoef(a,b)[0,1]),
            'rmse_over_reference_std':float(np.sqrt(np.mean(d*d))/max(np.std(b),1e-300))}
def boundary(root,label):
    stat=os.statvfs(root);free=stat.f_bavail*stat.f_frsize
    own=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    selected=[r for r in rows if r.split(',')[0].strip()==TARGET_UUID]
    snap={'at':now(),'label':label,'target_gpu':3,'uuid':TARGET_UUID,'target_contexts':selected,'free_bytes':free,'batch_bytes':own}
    save(root/'boundaries'/f'{label}.json',snap)
    if free<MIN_FREE or own>=LIMIT_BYTES:raise RuntimeError('storage guard paused '+label)
    if selected:raise RuntimeError('target GPU3 has another compute context; paused '+label)
    return snap
def verify_manifest(folder):
    d=json.loads((folder/'case_manifest.json').read_text())
    cleanup=folder.parent.parent/'cleanup_manifest.json'
    removed={x['path']:x for x in json.loads(cleanup.read_text())['removed_after_hash_and_science_validation']} if cleanup.exists() else {}
    for x in d['files']:
        p=folder/x['path']
        rel=str(p.relative_to(folder.parent.parent))
        if not p.exists() and rel in removed and removed[rel]['sha256']==x['sha256'] and removed[rel]['bytes']==x['bytes']:continue
        if not p.exists() or p.stat().st_size!=x['bytes'] or sha(p)!=x['sha256']:raise RuntimeError('resume hash failure '+str(p))
    return json.loads((folder/'metrics.json').read_text())
def analysis(folder,segments,compute_w):
    data=list(csv.DictReader((folder/'cpu_oracle.csv').open()))
    ix=np.fromfile(folder/'probe_sample_indices.u32',dtype='<u4')
    ids=np.fromfile(folder/'ids.u32',dtype='<u4')
    assert ix.shape==(16,) and ids.shape==(8192,)
    execution=json.loads((folder/'execution_metrics.json').read_text())
    assert execution['rows']==195468*segments
    checks=[];wchecks=[];physical=[]
    for mode in (0,1):
        n=np.fromfile(folder/f'raw_N_mode{mode}.f32',dtype='<f4');assert n.shape==(8192,) and np.isfinite(n).all()
        w=np.fromfile(folder/f'W_mode{mode}.f64',dtype='<f8') if compute_w else None
        if w is not None:assert w.shape==(8192,) and np.isfinite(w).all() and np.all(w>=0)
        for v in range(7):
            r=sorted([x for x in data if int(x['mode'])==mode and int(x['variant'])==v],key=lambda x:int(x['probe']))
            assert len(r)==16
            counts=np.array([int(x['count']) for x in r]);gcounts=np.array([int(x['gpu_aux_count']) for x in r]);assert np.all(counts>0)
            e=compare(n[ix],[float(x['N']) for x in r])
            item={'mode':mode,'variant':v,'counts_equal':bool(np.array_equal(counts,gcounts)),'min_probe_count':int(counts.min()),**e,'pass':bool(e['relative_l2']<=1e-3 and e['max_abs_over_peak']<=1e-3)}
            if v<6:
                checks.append(item)
                if w is not None:
                    wr=np.array([float(x['W']) for x in r]);we=compare(w[ix],wr)
                    wchecks.append({'mode':mode,'variant':v,'counts_equal':item['counts_equal'],**we,'pass':bool(we['relative_l2']<=1e-10 and item['counts_equal'])})
            else:physical.append(item)
    maskok=execution['point_mask_disagreements_outside_proved_bound']==0 and execution['max_unit_norm_deviation']<=2e-6
    passed=all(x['pass'] for x in checks) and all(x['pass'] for x in wchecks) and maskok
    return {'state':'completed','gate_pass':passed,'segments':segments,'rows':execution['rows'],'rawN_checks_all_six_FP32_mask_orders':checks,'W_checks_all_six_FP32_mask_orders':wchecks,'physical_FP64_dot_observation_not_a_tight_sensitivity_bound':physical,'mask_norm_gate_pass':maskok,'execution':execution,'scope':'8192 fixed output directions, complete cached half-baselines for stated segments, same consumed-n V, no new forward, no full map, no timing claim'}
def case(root,name,binary,path,segments=1,compute_w=False):
    final=root/'runs'/name
    if final.exists():return verify_manifest(final)
    boundary(root,'before_'+name)
    temp=root/'attempts'/(name+'_'+datetime.datetime.now().strftime('%H%M%S'))
    temp.mkdir(parents=True,exist_ok=False)
    replay=root.parent/'T01_20261101_production_replay'
    command=[str(root/'bin'/binary),'--gpus=3','--cache_root='+str(replay/'cache'),'--case_dir='+str(replay/'runs/day1_s10_g1_n8192'),'--dcf_bin=/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin','--out_dir='+str(temp),'--max_segments='+str(segments),'--replay_path='+path,'--compute_W='+str(int(compute_w))]
    save(temp/'command.json',{'at':now(),'command':command,'binary_sha256':sha(root/'bin'/binary),'environment':{'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'8'},'purpose':'numerical gate only'})
    save(root/'status.json',{'state':'running','case':name,'at':now(),'scope':'cached-V inverse only'})
    with (temp/'run.log').open('wb') as f:
        p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1'})
        try:rc=p.wait(timeout=900)
        except BaseException:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    log=(temp/'run.log').read_text()
    if rc or log.count('LEGACY_FASTPATH_SEG_DONE ')!=segments or log.count('LEGACY_FASTPATH_REPLAY_DONE')!=1:
        save(temp/'failed_execution.json',{'rc':rc,'at':now()});raise RuntimeError('failed execution '+name+' preserved '+str(temp))
    metric=analysis(temp,segments,compute_w);save(temp/'metrics.json',metric)
    files=[{'path':str(p.relative_to(temp)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(temp.rglob('*')) if p.is_file()]
    save(temp/'case_manifest.json',{'at':now(),'files':files,'name':name,'output_directory_at_execution':str(temp)})
    final.parent.mkdir(exist_ok=True);os.replace(temp,final)
    boundary(root,'after_'+name)
    print(json.dumps({'case':name,'gate_pass':metric['gate_pass'],'at':now()}),flush=True)
    return metric
def full_comparisons(root):
    replay=root.parent/'T01_20261101_production_replay'
    d=np.load(replay/'runs/day1_s10_g1_n8192/case.npz')
    oldroot=root.parent/'T01_20261101_original3d_replay'
    rep=np.divide(d['rep_N'],d['rep_W'],out=np.zeros_like(d['rep_N'],dtype='f8'),where=d['rep_W']>0)
    result={}
    for label in ('old','fixed'):
        folder=root/'runs'/('day1_direct_'+label);ids=np.fromfile(folder/'ids.u32',dtype='<u4');assert np.array_equal(ids,d['pixel_ids'])
        result[label]={}
        for mode in (0,1):
            n=np.fromfile(folder/f'raw_N_mode{mode}.f32',dtype='<f4').astype('f8');w=np.fromfile(folder/f'W_mode{mode}.f64',dtype='<f8');valid=w>0
            c=np.divide(n,w,out=np.zeros_like(n),where=valid)
            control=np.fromfile(oldroot/'runs'/label/f'raw_N_mode{mode}.f32',dtype='<f4')
            result[label][str(mode)]={'normalization':'diagnostic C=N/W; original optimized3D software emits rawN','reference_valid_pixels':int(valid.sum()),'candidate_zero_W_on_reference_domain':int(np.count_nonzero(valid&(d['rep_W']==0))),'STS_vs_normalized_3D':compare(rep[valid],c[valid]),'optimized_vs_controlled_sameV_rawN':compare(n,control),'W_min':float(w.min()),'W_max':float(w.max())}
    save(root/'metrics/full_day_comparisons.json',result)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);a=p.parse_args();root=a.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(root/'execution_protocol.json',{'at':now(),'target_gpu':3,'target_uuid':TARGET_UUID,'N_relative_l2_max_over_peak_limit':1e-3,'W_relative_l2_limit':1e-10,'max_batch_bytes':LIMIT_BYTES,'min_free_bytes':MIN_FREE,'original_parallel_repeats':3,'atomic_repeats_if_parallel_fails':3,'formal_timing':False,'new_forward':False,'450day':False})
        compile_data=json.loads((root/'compile_manifest.json').read_text())
        for x in compile_data['files']:
            target=root/x['path'];assert target.stat().st_size==x['bytes'] and sha(target)==x['sha256'],str(target)
        results={}
        for name,binname,path in [('seg1_direct_old','replay_legacy_fastpath_oldlookup','direct'),('seg1_direct_fixed','replay_legacy_fastpath_fixedlookup','direct'),('seg1_single_old','replay_legacy_fastpath_oldlookup','tasklist_single')]:
            results[name]=case(root,name,binname,path,compute_w=True)
            if not results[name]['gate_pass']:raise RuntimeError('control gate failed '+name)
        failed=False
        for j in range(1,4):
            name=f'seg1_parallel_old_r{j}';results[name]=case(root,name,'replay_legacy_fastpath_oldlookup','tasklist_parallel');failed|=not results[name]['gate_pass']
        if failed:
            for j in range(1,4):
                name=f'seg1_parallel_atomic_r{j}';results[name]=case(root,name,'replay_legacy_fastpath_atomic_oldlookup','tasklist_parallel')
        for label in ('old','fixed'):
            name='day1_direct_'+label;results[name]=case(root,name,'replay_legacy_fastpath_'+('oldlookup' if label=='old' else 'fixedlookup'),'direct',segments=10,compute_w=True)
            if not results[name]['gate_pass']:raise RuntimeError('full-day inverse gate failed '+name)
        comparison=full_comparisons(root)
        final={'state':'completed','at':now(),'cases':{k:{'gate_pass':v['gate_pass'],'segments':v['segments']} for k,v in results.items()},'original_parallel_any_failed':failed,'full_day_comparisons':comparison,'scope':'numerical controls only; original planner performance and historical full-image correctness not established by small-output race protocol','all_valid_repeats_retained':True}
        save(root/'summary.json',final);save(root/'status.json',{'state':'completed','at':now(),'new_forward':False,'full_map':False,'450day':False,'summary':'summary.json'})
        print(json.dumps({'state':'completed','at':now(),'cases':len(results),'original_parallel_any_failed':failed}),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        root=Path(__file__).resolve().parent.parent
        save(root/'status.json',{'state':'paused_or_failed','at':now(),'error':str(e),'attempts_preserved':True});raise
