#!/usr/bin/env python3
import argparse,csv,datetime,fcntl,hashlib,json,os,re,signal,subprocess,sys
from pathlib import Path
import numpy as np

UUID='GPU-99fbd651-f080-aa22-391b-f33caa723578'
METHODS={'A':('replay_controlled_fixed_full',[]),'B':('replay_legacy_combined_full',[]),'C':('replay_sts_full',['--rep_phase=grid']),'D':('replay_sts_full',['--rep_phase=actual'])}
BLOCKS=[['A','B','D','C'],['B','C','A','D'],['C','D','B','A'],['D','A','C','B']]
def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
def save(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def cmp(a,b):
    a=np.asarray(a,dtype='f8');b=np.asarray(b,dtype='f8');d=a-b
    return {'relative_l2':float(np.linalg.norm(d)/max(np.linalg.norm(b),1e-300)),'max_abs_over_peak':float(np.max(abs(d))/max(np.max(abs(b)),1e-300)),'bitwise_equal':bool(np.array_equal(a,b))}
def boundary(root,label):
    if (root/'PAUSE').exists():raise RuntimeError('PAUSE requested at boundary')
    fs=os.statvfs(root);free=fs.f_bavail*fs.f_frsize;own=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    target=[x for x in rows if x.split(',')[0].strip()==UUID]
    snap={'at':now(),'label':label,'target_gpu':3,'target_uuid':UUID,'target_contexts':target,'free_bytes':free,'own_batch_bytes':own}
    save(root/'boundaries'/(label+'.json'),snap)
    if target:raise RuntimeError('GPU3 busy, pause without waiting for a card')
    if free<100*1024**3 or own>=100*1024**2:raise RuntimeError('storage guard paused')
def inputs(root):
    common=root.parent.parent/'01_common_operator';replay=common/'T01_20261101_production_replay'
    return common,replay
def freeze(root):
    common,replay=inputs(root);files=[]
    selected=[p for folder in ('source','scripts','bin') for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    selected += [replay/'source'/name for name in ('common_mirror_tile_vissfast.hpp','orbit_online_mirror_tile_vissfast.cuh','viss_recon_kernel_mirror_tile_vissfast.cuh')]+[replay/'scripts/replay_common.hpp']
    selected += sorted((replay/'cache').glob('day1_seg*.bin'))
    selected += [root/'PROTOCOL.md',common/'T01_20261101_legacy3d_lineage/runs/day1_direct_fixed/raw_N_mode1.f32',replay/'runs/day1_s10_g1_n8192/case.npz',common/'T21_20261101_rep_phase_correction/runs/day1/candidate.npz']
    for p in selected:files.append({'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)})
    dst=root/'freeze.json'
    if dst.exists():
        old=json.loads(dst.read_text());oldmap={x['path']:x for x in old['files']}
        for x in files:
            if x['path'] in oldmap and x!=oldmap[x['path']]:raise RuntimeError('frozen identity changed '+x['path'])
    else:save(dst,{'at':now(),'scope':'source/binary/cache/reference SHA before functional gates','files':files,'formal_blocks':BLOCKS,'warmups':list(METHODS),'thresholds':{'A_B_D_N_l2_and_overpeak':1e-3,'C_N_bitwise':True,'C_D_W_exact':True}})
def scientific_check(root,folder,method):
    common,replay=inputs(root);d=np.load(replay/'runs/day1_s10_g1_n8192/case.npz');ids=np.fromfile(folder/'sample_ids.u32',dtype='<u4');n=np.fromfile(folder/'sample_N.f32',dtype='<f4')
    if not np.array_equal(ids,d['pixel_ids']) or n.shape!=(8192,) or not np.isfinite(n).all():raise RuntimeError('sample IDs/finite failed')
    if method in ('A','B'):ref=np.fromfile(common/'T01_20261101_legacy3d_lineage/runs/day1_direct_fixed/raw_N_mode1.f32',dtype='<f4')
    elif method=='C':ref=np.asarray(d['rep_N'],dtype='<f4')
    else:
        t=np.load(common/'T21_20261101_rep_phase_correction/runs/day1/candidate.npz');assert np.array_equal(ids,t['pixel_ids']);ref=np.asarray(t['corrected_mid_N'],dtype='<f4')
    error=cmp(n,ref);passed=error['bitwise_equal'] if method=='C' else error['relative_l2']<=1e-3 and error['max_abs_over_peak']<=1e-3
    wcheck=None
    if method in ('C','D'):
        w=np.fromfile(folder/'sample_W.u32',dtype='<u4');wr=d['rep_W'] if method=='C' else t['corrected_mid_W'];wcheck=bool(np.array_equal(w,wr));passed &=wcheck
    stats=json.loads((folder/'full_stats.json').read_text());raw=stats['rawN'];all_finite=(raw.get('finite_count',201326592-raw.get('nonfinite',0))==201326592 and raw['count']==201326592)
    passed &=all_finite and bool(re.fullmatch('[0-9a-f]{64}',raw['sha256']))
    return {'pass':bool(passed),'method':method,'sample_N':error,'sample_W_exact':wcheck,'all201326592_finite':bool(all_finite),'reference':'same cachedV original frozen production or fixedDCF/T21 sample; no affine calibration','full_rawN_sha256':raw['sha256']}
def verify(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for x in manifest['files']:
        p=folder/x['path'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],str(p)
    return json.loads((folder/'result.json').read_text())
def run(root,name,method,category):
    final=root/'runs'/name
    if final.exists():return verify(final)
    boundary(root,'before_'+name);temp=root/'attempts'/(name+'_'+datetime.datetime.now().strftime('%H%M%S'));temp.mkdir(parents=True,exist_ok=False)
    common,replay=inputs(root);binary,extra=METHODS[method]
    command=[str(root/'bin'/binary),'--gpus=3','--cache_root='+str(replay/'cache'),'--out_dir='+str(temp),'--nside=4096','--day=1','--max_segments=10']+extra
    save(temp/'command.json',{'at':now(),'command':command,'binary_sha256':sha(root/'bin'/binary),'category':category,'method':method,'CUDA_DEVICE_ORDER':'PCI_BUS_ID'})
    save(root/'status.json',{'state':'running','at':now(),'case':name,'category':category,'method':method,'full_output_pixels':201326592})
    with (temp/'run.log').open('wb') as f:
        child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'CUDA_DEVICE_ORDER':'PCI_BUS_ID','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        try:rc=child.wait(timeout=7200)
        except BaseException:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            raise
    log=(temp/'run.log').read_text();token='FULL_INVERSE_DONE' if method in ('A','B') else 'FULL_STS_DONE'
    if rc or log.count(token)!=1:raise RuntimeError('execution failed '+name+' preserved '+str(temp))
    metrics=json.loads((temp/'metrics.json').read_text());assert metrics['inverse_loop_wall_s']>0
    rows=list(csv.DictReader((temp/'segments.csv').open()));assert len(rows)==10
    check=scientific_check(root,temp,method);result={'state':'completed','at':now(),'category':category,'method':method,'name':name,'gate':check,'timing':metrics}
    save(temp/'result.json',result)
    files=[{'path':str(p.relative_to(temp)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(temp.rglob('*')) if p.is_file()]
    save(temp/'manifest.json',{'at':now(),'files':files,'output_directory_at_execution':str(temp)})
    final.parent.mkdir(exist_ok=True);os.replace(temp,final);boundary(root,'after_'+name)
    print(json.dumps({'at':now(),'name':name,'gate':check['pass'],'inverse_loop_wall_s':metrics['inverse_loop_wall_s']}),flush=True)
    if not check['pass']:raise RuntimeError('scientific gate failed '+name+'; preserved, no formal expansion')
    return result
def summarize(root):
    result={}
    for method in METHODS:
        selected=[json.loads(p.read_text()) for p in (root/'runs').glob('formal_*/result.json') if json.loads(p.read_text())['method']==method]
        values=[x['timing']['inverse_loop_wall_s'] for x in selected]
        if values:
            a=np.array(values);result[method]={'raw_inverse_loop_wall_s':values,'n':len(values),'median_s':float(np.median(a)),'mean_s':float(a.mean()),'sd_s':float(a.std(ddof=1)) if len(a)>1 else None,'iqr_s':float(np.quantile(a,.75)-np.quantile(a,.25)),'all_numerical_gates_pass':all(x['gate']['pass'] for x in selected)}
    save(root/'summary.json',{'state':'completed' if all(x.get('n')==4 for x in result.values()) and len(result)==4 else 'partial','at':now(),'methods':result,'formal_blocks':BLOCKS,'all_valid_values_retained':True,'scope':'one physical4090, full201M output, cached inverse only; scientific weights differ; no forward/full-output file/450day'})
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=('gates','formal'),required=True);a=ap.parse_args();root=Path(__file__).resolve().parent.parent
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);freeze(root)
        if a.phase=='gates':
            for method in METHODS:run(root,'gate_'+method,method,'functional_gate')
            save(root/'status.json',{'state':'all_four_gates_passed_waiting_formal_coordination','at':now(),'formal_started':False});print('ALL_FOUR_GATES_PASSED',flush=True)
        else:
            for method in METHODS:
                gate=verify(root/'runs'/('gate_'+method));assert gate['gate']['pass']
            if not (root/'FORMAL_ALLOWED').exists():raise RuntimeError('root-agent formal coordination marker required')
            subprocess.run(['tmux','wait-for','-S','legacy_inverse_formal_started_20261101'],check=True)
            for method in METHODS:run(root,'warmup_'+method,method,'warmup')
            had_formal=bool(list((root/'runs').glob('formal_*')));missing=any(not (root/'runs'/f'formal_b{i}_{m}').exists() for i,b in enumerate(BLOCKS,1) for m in b)
            if had_formal and missing:
                stamp=datetime.datetime.now().strftime('%H%M%S')
                for method in METHODS:run(root,'resume_warmup_'+stamp+'_'+method,method,'extra_resume_warmup')
            for i,block in enumerate(BLOCKS,1):
                for method in block:run(root,f'formal_b{i}_{method}',method,'formal');summarize(root)
            summarize(root);save(root/'status.json',{'state':'completed','at':now(),'gates':4,'warmups':4,'formal':16,'all_valid_formal_values_retained':True});print('FORMAL_COMPLETED',flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(Path(__file__).resolve().parent.parent/'status.json',{'state':'paused_or_failed','at':now(),'error':str(e),'attempts_preserved':True});raise
