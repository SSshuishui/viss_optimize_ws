#!/usr/bin/env python3
"""One-shot dependency wait and GPU2 execution; completion signal on all exits."""
import datetime,json,math,os,shutil,subprocess,sys,traceback,uuid
from pathlib import Path
P=Path(__file__).resolve().parents[1]
DEPENDENCY=P.parents[1]/'99_handoffs/non450_20261101/status.json'
S=P/'chain_state.json'
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(v):
    t=S.with_name(S.name+'.tmp.'+uuid.uuid4().hex)
    with t.open('w') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(t,S)
state={'state':'waiting','created_at':stamp(),'wait_signal':'parco_non450_done','done_signal':'parco_roi_v2_done','dependency_status_path':str(DEPENDENCY),'GPU_physical':2,'retry_policy':'once; no automatic retry'}
def command(args):return subprocess.check_output(args,text=True).strip()
def finite(v):
    if isinstance(v,dict):return all(finite(q) for q in v.values())
    if isinstance(v,list):return all(finite(q) for q in v)
    return math.isfinite(v) if isinstance(v,(int,float)) and not isinstance(v,bool) else True
rc=1
try:
    save(state)
    subprocess.run(['tmux','wait-for','parco_non450_done'],check=True)
    dependency=json.loads(DEPENDENCY.read_text());state['dependency_state']=dependency.get('state');state['dependency_checked_at']=stamp()
    if dependency.get('state')!='completed':raise RuntimeError('dependency status is not completed; no GPU execution')
    free=shutil.disk_usage(P).free
    state['storage_free_bytes']=free
    if free<100*1024**3:raise RuntimeError('storage guard: less than100GiB free')
    gpu=command(['nvidia-smi','--id=2','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader,nounits'])
    parts=[v.strip() for v in gpu.split(',')];gpu_uuid=parts[0];used=float(parts[1]);total=float(parts[2])
    processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'])
    owned=[line for line in processes.splitlines() if line.split(',')[0].strip()==gpu_uuid]
    state['one_time_preflight']={'checked_at':stamp(),'uuid':gpu_uuid,'memory_used_MiB':used,'memory_total_MiB':total,'compute_contexts':owned,'memory_idle_limit_MiB':256,'utilization_not_polled':True}
    if owned or used>256:raise RuntimeError('GPU2 context/memory guard found busy device; no execution')
    target=P/'metrics/gpu_roi.json'
    if target.exists():raise RuntimeError('prior v2 metrics exist; refuse automatic overwrite/retry')
    state['state']='running';state['started_at']=stamp();save(state)
    args=['timeout','100s','env','CUDA_VISIBLE_DEVICES=2','python3','-u',str(P/'scripts/gpu_roi_v2.py')]
    with (P/'run.log').open('w') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
    state['returncode']=r.returncode
    if r.returncode!=0:raise RuntimeError('GPU child returned nonzero: '+str(r.returncode))
    result=json.loads(target.read_text())
    if result.get('status')!='completed' or not finite(result):raise RuntimeError('result status/finite validation failed')
    for eps in ('1e-06','1e-09'):
        q=result['results'][eps]
        if q['apply_wall']['n']!=10 or not math.isfinite(q['error_vs_CPU']['relative_l2']) or not math.isfinite(q['normalized_dirty_error']['relative_l2']):raise RuntimeError('repeat/error validation failed')
    if result['GPU_direct_sameprocess']['apply_wall']['n']!=10:raise RuntimeError('direct repeat validation failed')
    state.update(state='completed',finished_at=stamp(),result_status=result['status'],finite_error_validation=True,GPU_process_exited=True)
    save(state);rc=0
except BaseException as exc:
    state.update(state='failed',finished_at=stamp(),error=repr(exc),traceback=traceback.format_exc(),GPU_process_exited=True)
    save(state);print(json.dumps(state,indent=2),flush=True)
finally:
    signal=subprocess.run(['tmux','wait-for','-S','parco_roi_v2_done'],capture_output=True,text=True)
    state['done_signal_returncode']=signal.returncode;save(state)
sys.exit(rc)
