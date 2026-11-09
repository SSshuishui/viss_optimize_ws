#!/usr/bin/env python3
import argparse,csv,datetime,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import numpy as np

UUID='GPU-99fbd651-f080-aa22-391b-f33caa723578'
BUDGET_LIMIT_S=900.0
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def compare(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');d=a-b
    return {'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-300)),'max_abs_over_peak':float(np.max(abs(d))/max(np.max(abs(b)),1e-300)),'max_abs':float(np.max(abs(d)))}
def boundary(root,name,planned_bytes=0):
    if (root/'PAUSE').exists():raise RuntimeError('PAUSE requested at boundary')
    fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize;own=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).strip().splitlines();target=[r for r in rows if r.split(',')[0].strip()==UUID]
    save(root/'boundaries'/(name+'.json'),{'at':now(),'target_uuid':UUID,'target_contexts':target,'free_bytes':free,'own_batch_bytes':own})
    if target:raise RuntimeError('GPU3 busy: pause without waiting for a card')
    if free<100*1024**3 or own+planned_bytes>=3*1024**3:raise RuntimeError('storage guard paused, including planned new output')
def require_formal_done(root):
    p=root.parent.parent/'07_performance_system/T07_20261101_legacy_inverse_comparison/status.json';s=json.loads(p.read_text())
    if s.get('state')!='completed' or s.get('formal')!=16:raise RuntimeError('formal is not completed/released; no GPU or compilation allowed')
    if not (root/'FULLREF_ALLOWED').is_file():raise RuntimeError('root-agent resource clearance/authorization marker FULLREF_ALLOWED required')
def process_identity(pid):
    try:
        stat=Path(f'/proc/{pid}/stat').read_text();rest=stat[stat.rfind(')')+2:].split();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if rest[0] in ('Z','X'):return None
        return {'pid':int(pid),'start_ticks':rest[19],'boot_id':boot}
    except (FileNotFoundError,PermissionError,IndexError):return None
def budget_load(root):
    p=root/'budget_state.json'
    if not p.exists():
        if any((root/name).is_dir() and any((root/name).iterdir()) for name in ('runs','attempts')):raise RuntimeError('budget state missing beside existing work; refuse automatic900s reset')
        state={'limit_seconds':BUDGET_LIMIT_S,'used_process_wall_seconds':0.0,'attempts':[],'accounting_scope':'each launched correctness/export process wall, including failed/timeout attempts; CPU validation/analysis separately timed'};save(p,state)
    else:state=json.loads(p.read_text())
    if state['limit_seconds']!=BUDGET_LIMIT_S:raise RuntimeError('persistent budget limit changed; explicit review required')
    if state.get('automatic_launch_blocked'):raise RuntimeError('unresolved launch identity consumed remaining budget; explicit audit required, no automatic reset')
    charged=sum(float(x.get('process_wall_seconds_charged',0)) for x in state['attempts'])
    if abs(charged-state['used_process_wall_seconds'])>1e-6:raise RuntimeError('persistent budget accounting mismatch; refuse automatic reset')
    active=[x for x in state['attempts'] if x['state'] in ('launching','running')]
    if active:
        for attempt in active:
            expected=attempt.get('process_identity');actual=process_identity(expected['pid']) if expected else None
            if expected is None:
                remaining=max(0.0,BUDGET_LIMIT_S-state['used_process_wall_seconds'])
                attempt.update({'state':'unresolved_launch_without_process_identity','finished_at':now(),'process_wall_seconds_charged':remaining,'wall_measurement':'cannot prove whether Popen launched child before parent interruption; conservatively consume all remaining budget','not_automatically_retried':True})
                state['used_process_wall_seconds']+=remaining;state['remaining_seconds']=0.0;state['automatic_launch_blocked']=True;save(p,state)
                raise RuntimeError('launch journal has no recorded process identity; all remaining budget charged, automatic launch blocked pending audit')
            if expected is not None and actual==expected:raise RuntimeError('previous owned attempt process still active; no duplicate GPU launch')
            remaining=max(0.0,BUDGET_LIMIT_S-state['used_process_wall_seconds']);delta=time.time()-attempt['started_epoch']
            charge=min(remaining,max(0.0,delta)) if delta>=0 else remaining
            attempt.update({'state':'interrupted_parent_or_restart_accounted','finished_at':now(),'process_wall_seconds_charged':charge,'wall_measurement':'conservative upper bound until recovery; exact child termination unavailable','not_automatically_retried':True})
            state['used_process_wall_seconds']+=charge
        save(p,state)
    return state
def budget_remaining(root):
    state=budget_load(root);return max(0.0,state['limit_seconds']-state['used_process_wall_seconds'])
def budget_begin(root,name,cmd):
    state=budget_load(root);remaining=max(0.0,state['limit_seconds']-state['used_process_wall_seconds'])
    if remaining<5.0:raise RuntimeError('persistent correctness process budget exhausted; no automatic reset')
    item={'attempt_number':len(state['attempts'])+1,'case':name,'command':cmd,'state':'launching','started_at':now(),'started_epoch':time.time(),'remaining_at_start_seconds':remaining}
    state['attempts'].append(item);save(root/'budget_state.json',state);return item['attempt_number'],remaining
def budget_set_pid(root,number,pid):
    state=json.loads((root/'budget_state.json').read_text());item=state['attempts'][number-1];item['state']='running';item['process_identity']=process_identity(pid);item['pid']=pid;save(root/'budget_state.json',state)
def budget_finish(root,number,wall,rc,error):
    state=json.loads((root/'budget_state.json').read_text());item=state['attempts'][number-1]
    item.update({'state':'finished' if error is None and rc==0 else 'failed_or_timeout','finished_at':now(),'process_wall_seconds_charged':wall,'wall_measurement':'measured monotonic process launch/wait/termination wall','returncode':rc,'error':error})
    state['used_process_wall_seconds']=sum(float(x.get('process_wall_seconds_charged',0)) for x in state['attempts']);state['remaining_seconds']=max(0.0,state['limit_seconds']-state['used_process_wall_seconds']);state['overage_seconds']=max(0.0,state['used_process_wall_seconds']-state['limit_seconds']);save(root/'budget_state.json',state)
def freeze(root):
    replay=root.parent/'T01_20261101_production_replay';old=root.parent/'T01_20261101_legacy3d_lineage/runs/day1_direct_fixed';formal=root.parent.parent/'07_performance_system/T07_20261101_legacy_inverse_comparison'
    selected=[p for folder in ('source','scripts','bin') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]+[root/'PROTOCOL.md']
    selected+=sorted((replay/'cache').glob('day1_seg*.bin'))
    selected+=[replay/'source'/n for n in ('common_mirror_tile_vissfast.hpp','orbit_online_mirror_tile_vissfast.cuh','viss_recon_kernel_mirror_tile_vissfast.cuh')]+[replay/'scripts/replay_common.hpp']
    selected+=[replay/'runs/day1_s10_g1_n8192'/n for n in ('ids.u32','l.f32','m.f32','n.f32')]
    selected+=[old/n for n in ('raw_N_mode1.f32','W_mode1.f64','probe_sample_indices.u32','ids.u32','cpu_oracle.csv')]
    selected+=[formal/'runs'/('gate_'+m)/n for m in 'CD' for n in ('sample_ids.u32','sample_N.f32','sample_W.u32','full_stats.json')]
    selected+=[Path('/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin')]
    files=[{'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)} for p in selected]
    dest=root/'freeze.json'
    if dest.exists():
        expected={x['path']:x for x in json.loads(dest.read_text())['files']}
        for x in files:
            if x['path'] in expected and expected[x['path']]!=x:raise RuntimeError('frozen input changed '+x['path'])
    else:save(dest,{'at':now(),'files':files,'scope':'source/bin/cache/DCF/reference identities before correctness GPU; formal unchanged'})
def verify_case(folder):
    meta=json.loads((folder/'manifest.json').read_text());removed={}
    cleanup=folder.parent.parent/'cleanup_manifest.json'
    if cleanup.exists():removed={x['path']:x for x in json.loads(cleanup.read_text())['removed_after_validation']}
    for x in meta['files']:
        p=folder/x['path'];rel=str(p.resolve())
        if not p.exists() and rel in removed and removed[rel]['sha256']==x['sha256']:continue
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
    return json.loads((folder/'gate.json').read_text())
def zero_check(folder,mode):
    nr=np.fromfile(folder/'N_real.f32',dtype='<f4');nz=np.fromfile(folder/'N_zero.f32',dtype='<f4');wr=np.fromfile(folder/'W_real.f64',dtype='<f8');wz=np.fromfile(folder/'W_zero.f64',dtype='<f8');d=json.loads((folder/'control.json').read_text())
    assert all(a.shape==(8192,) and np.isfinite(a).all() for a in (nr,nz,wr,wz));assert d['OP_RECON_SKIP_ZERO_WEIGHTED_VISS']==1 and d['OP_RECON_DEBUG_CHECK_FINITE']==1
    error=compare(wz,wr);ok=np.count_nonzero(nz)==0 and np.all(wr>0) and error['relative_l2']<=1e-10;uniform=None
    if mode=='all_visible':uniform=compare(wr,np.full(8192,d['q_sum_independent_CPU']));ok &=uniform['relative_l2']<=1e-10
    return {'pass':bool(ok),'zero_N_nonzero_count':int(np.count_nonzero(nz)),'W_zeroV_vs_realV':error,'uniform_vs_CPU_qsum':uniform,'mask_mode':mode,'visibility_skip_guard_enabled':True,'scope':'firstsegment8192, W independent ofV; point mode and artificial all-visible uniformW control separate'}
def reference_check(root,folder):
    old=root.parent/'T01_20261101_legacy3d_lineage/runs/day1_direct_fixed'
    ix=np.fromfile(old/'probe_sample_indices.u32',dtype='<u4');refN=np.fromfile(old/'raw_N_mode1.f32',dtype='<f4');refW=np.fromfile(old/'W_mode1.f64',dtype='<f8')
    n=np.fromfile(folder/'sample_N.f32',dtype='<f4');w=np.fromfile(folder/'sample_W.f64',dtype='<f8');ids=np.fromfile(folder/'sample_ids.u32',dtype='<u4');oldids=np.fromfile(old/'ids.u32',dtype='<u4');assert np.array_equal(ids,oldids) and n.shape==w.shape==(8192,) and np.isfinite(n).all() and np.isfinite(w).all() and np.all(w>=0)
    en=compare(n,refN);ew=compare(w,refW);checks=[]
    rows=list(csv.DictReader((old/'cpu_oracle.csv').open()))
    for variant in range(6):
        r=sorted([x for x in rows if int(x['mode'])==1 and int(x['variant'])==variant],key=lambda x:int(x['probe']));assert len(r)==16 and all(int(x['count'])>0 for x in r)
        a=compare(n[ix],[float(x['N']) for x in r]);b=compare(w[ix],[float(x['W']) for x in r]);checks.append({'CPU_FMA_variant':variant,'min_reference_nonempty_count':min(int(x['count']) for x in r),'N':a,'W':b,'pass':a['relative_l2']<=1e-3 and a['max_abs_over_peak']<=1e-3 and b['relative_l2']<=1e-10})
    stats=json.loads((folder/'full_stats.json').read_text());assert stats['rawN']['count']==stats['rawN']['finite_count']==201326592 and stats['W']['count']==stats['W']['finite_count']==201326592 and stats['C']['count']==stats['C']['finite_count']==201326592
    assert (folder/'corrected_dcf_C.f32').stat().st_size==805306368 and (folder/'native_regions.u8').stat().st_size==201326592
    assert sha(folder/'corrected_dcf_C.f32')==stats['C']['sha256'] and sha(folder/'native_regions.u8')==stats['native_regions']['sha256']
    raw=json.loads((folder/'raw_output_full_metrics.json').read_text());assert raw['reference_W0_count']==stats['W']['zero_count']
    for method in ('original_STS_grid_C','actual_uv_STS_C'):
        for domain,count in [('all201M_including_W0',201326592),('secondary_reference_W_gt0',201326592-stats['W']['zero_count'])]:
            values=raw['metrics'][method][domain]['regions'];assert values['full_domain']['count']==count and sum(values[key]['count'] for key in ('native_abs_lat_le30','native_abs_lat_30to60','native_abs_lat_gt60'))==count
    return {'pass':bool(en['relative_l2']<=1e-3 and en['max_abs_over_peak']<=1e-3 and all(x['pass'] for x in checks)),'all8192_N_vs_accepted_fixedDCF':en,'all8192_W_difference_diagnostic':ew,'independent_CPU16_all_six_FP32_masks':checks,'full201M_finite':True,'full_C_and_tag_file_SHA_verified':True,'normalization':'explicit diagnostic N_F32/W_F64, original rawN retained in fullSHA andsamples','all8192_W_scope':'record all differences; strict1e-10 is predeclared for16 independent CPU probes, not silently applied to native cone versus point boundaries over all8192'}
def sts_check(root,folder,phase):
    formal=root.parent.parent/'07_performance_system/T07_20261101_legacy_inverse_comparison';label='C' if phase=='grid' else 'D';old=formal/'runs'/('gate_'+label)
    equal={name:(folder/name).read_bytes()==(old/name).read_bytes() for name in ('sample_ids.u32','sample_N.f32','sample_W.u32')}
    d=json.loads((folder/'full_stats.json').read_text());ref=json.loads((old/'full_stats.json').read_text());same=d['C']['sha256']==ref['C']['sha256'];p=folder/('sts_'+phase+'_C.f32');assert p.stat().st_size==805306368;assert sha(p)==d['C']['sha256']
    return {'pass':bool(all(equal.values()) and same),'sample_bytes_equal':equal,'full_C_SHA_equals_accepted_formal_gate':same,'full_C_SHA':d['C']['sha256'],'export_file_SHA_verified':True,'phase':phase}
def case(root,name,binary,extra):
    final=root/'runs'/name
    if final.exists():
        old=verify_case(final)
        if not old['pass']:raise RuntimeError('preserved numerical failure '+name+' requires diagnosis/new candidate; no automatic expansion')
        return old
    unresolved=[p for p in (root/'attempts').glob(name+'_*') if p.is_dir()]
    if unresolved:raise RuntimeError('uncommitted prior attempt for '+name+' requires explicit audit; preserved outputs are not automatically retried: '+str(unresolved))
    if budget_remaining(root)<5:raise RuntimeError('persistent900s correctness process budget exhausted before '+name)
    planned=1000000 if name.startswith('zero_') else (201326592*(5 if name=='reference' else 4)+1000000)
    boundary(root,'before_'+name,planned);temp=root/'attempts'/(name+'_'+datetime.datetime.now().strftime('%H%M%S'));temp.mkdir(parents=True,exist_ok=False)
    replay=root.parent/'T01_20261101_production_replay';cmd=[str(root/'bin'/binary),'--gpus=3','--cache_root='+str(replay/'cache'),'--out_dir='+str(temp),'--nside=4096','--day=1','--max_segments=10']+extra
    if binary=='zero_v_control':cmd+=['--sample_dir='+str(replay/'runs/day1_s10_g1_n8192')]
    if name=='reference':
        for oldname in ('sts_grid','sts_actual'):assert verify_case(root/'runs'/oldname)['pass']
        cmd+=['--sts_grid_C='+str(root/'runs/sts_grid/sts_grid_C.f32'),'--sts_actual_C='+str(root/'runs/sts_actual/sts_actual_C.f32')]
    save(temp/'command.json',{'at':now(),'command':cmd,'binary_sha256':sha(root/'bin'/binary),'purpose':'correctness, never formal performance','CUDA_DEVICE_ORDER':'PCI_BUS_ID'})
    save(root/'status.json',{'state':'running','at':now(),'case':name,'correctness_only':True})
    number,remaining=budget_begin(root,name,cmd);began=time.monotonic();child=None;rc=None;error=None
    try:
        with (temp/'run.log').open('wb') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'8'});budget_set_pid(root,number,child.pid)
            rc=child.wait(timeout=max(.1,remaining-5.0))
    except BaseException as exc:
        error=repr(exc)
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=4)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if child is not None:rc=child.returncode
        raise
    finally:budget_finish(root,number,time.monotonic()-began,rc,error)
    if rc:raise RuntimeError(name+' execution failed; outputs preserved at '+str(temp))
    if name.startswith('zero_'):gate=zero_check(temp,name.removeprefix('zero_'))
    elif name=='reference':gate=reference_check(root,temp)
    else:gate=sts_check(root,temp,'grid' if name=='sts_grid' else 'actual')
    save(temp/'gate.json',gate);files=[{'path':str(p.relative_to(temp)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(temp.rglob('*')) if p.is_file()]
    save(temp/'manifest.json',{'at':now(),'files':files,'output_directory_at_execution':str(temp)})
    final.parent.mkdir(exist_ok=True);os.replace(temp,final);boundary(root,'after_'+name);state=budget_load(root);print(json.dumps({'case':name,'pass':gate['pass'],'at':now(),'persistent_process_wall_used_s':state['used_process_wall_seconds'],'remaining_budget_s':budget_remaining(root)}),flush=True)
    if not gate['pass']:raise RuntimeError(name+' numerical gate failed, preserved; no automatic expansion')
    return gate
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();root=Path(__file__).resolve().parent.parent
    if not a.execute:print('Prepared only. --execute requires the separate formal batch completed/released.');return
    require_formal_done(root)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze(root)
        if (root/'cleanup_manifest.json').exists() and (root/'full_domain_metrics.json').exists():
            completed=json.loads((root/'full_domain_metrics.json').read_text());parent=root/'metrics/parent_nside128.npz';assert completed['state']=='completed' and parent.stat().st_size==completed['parent_artifact_bytes'] and sha(parent)==completed['parent_artifact_sha256']
            cleanup=json.loads((root/'cleanup_manifest.json').read_text())
            for item in cleanup['removed_after_validation']:
                p=Path(item['path']).resolve();assert p.is_relative_to(root/'runs') and p.name in ('corrected_dcf_C.f32','sts_grid_C.f32','sts_actual_C.f32','native_regions.u8')
                if p.exists():assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];p.unlink()
            for folder in (root/'runs').iterdir():
                if folder.is_dir():assert verify_case(folder)['pass']
            for item in completed['temporary_artifacts']:item['state']='removed_after_validation'
            save(root/'full_domain_metrics.json',completed);save(root/'status.json',{'state':'completed','at':now(),'resume_finalization_only':True,'temporary_maps_cleaned':True,'no_GPU_rerun':True});print('ALREADY_COMPLETED_VALIDATED_NO_GPU_RERUN',flush=True);return
        for folder in sorted((root/'runs').iterdir()) if (root/'runs').exists() else []:
            if folder.is_dir() and not verify_case(folder)['pass']:raise RuntimeError('preserved failed case requires diagnosis; no restart expansion')
        budget_load(root);start=time.monotonic();gates={}
        for name,binary,extra in [('zero_point','zero_v_control',['--mask=point']),('zero_all_visible','zero_v_control',['--mask=all_visible']),('sts_grid','replay_sts_export',['--rep_phase=grid']),('sts_actual','replay_sts_export',['--rep_phase=actual']),('reference','replay_corrected_dcf_full',[])]:gates[name]=case(root,name,binary,extra)
        save(root/'GPU_gates_summary.json',{'state':'completed','at':now(),'gates':gates,'queue_plus_validation_wall_this_invocation_s':time.monotonic()-start,'persistent_budget_state':budget_load(root),'no_performance_claim':True})
        with (root/'CPU_analysis.log').open('wb') as f:subprocess.run([sys.executable,str(root/'scripts/analyze_full_maps.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
        metrics=json.loads((root/'full_domain_metrics.json').read_text());removed=[]
        for item in metrics['temporary_artifacts']:
            p=Path(item['path']).resolve();assert p.is_relative_to(root/'runs') and p.name in ('corrected_dcf_C.f32','sts_grid_C.f32','sts_actual_C.f32','native_regions.u8');assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'];removed.append({**item,'state':'removed_after_full_SHA_metrics_parent_QA_validation'})
        save(root/'cleanup_manifest.json',{'at':now(),'removed_after_validation':removed,'scope':'only this batch 3C andregion tags; old maps untouched'})
        for item in removed:Path(item['path']).unlink()
        for item in metrics['temporary_artifacts']:item['state']='removed_after_validation'
        save(root/'full_domain_metrics.json',metrics);own=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert own<100*1024**2
        save(root/'status.json',{'state':'completed','at':now(),'all5_correctness_gates_passed':True,'full_domain_metrics':'full_domain_metrics.json','own_final_bytes':own,'temporary_maps_cleaned':True,'no_new_forward_no450':True});print('FULL_CORRECTED_DCF_REFERENCE_COMPLETED',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(Path(__file__).resolve().parent.parent/'status.json',{'state':'paused_or_failed','at':now(),'error':str(e),'failed_outputs_preserved':True});raise
