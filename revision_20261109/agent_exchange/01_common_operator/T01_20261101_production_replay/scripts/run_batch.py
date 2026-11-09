#!/usr/bin/env python3
"""Sequential correctness replay on physical GPU3, independent resumable cases."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

BASE = Path('/data/zhaox/research/viss_optimize_ws')
class GpuBusy(RuntimeError):pass

def stamp():return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:json.dump(data,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def idle_snapshot():
    # Memory/process state is current; utilization may still reflect our previous completed run.
    result=subprocess.run(['nvidia-smi','--id=3','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=20)
    fields=next(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    if int(fields[0])!=3 or float(fields[2])>512:raise GpuBusy('GPU3 not idle at case boundary: '+result.stdout.strip())
    return {'raw':result.stdout.strip(),'loadavg':os.getloadavg(),'at':stamp()}
def tasks(root,phase):
    cache=root/'cache';bin_dir=root/'bin';dcf=BASE/'system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin'
    out=[]
    def cache_task(day,gate=False):
        cmd=[str(bin_dir/'main_cache'),'--gpus=3','--btag=10M','--nside=4096',f'--day_start={day}','--day_count=1','--segs=10','--occ_mode=2','--blockage=1','--orbit_seed=42','--grid_cell_scale=1','--B_mode=bin','--C_mode=bin',f'--sky_dir={BASE}/earth_10Mhz',f'--out_dir={root}/cache_profiles/day{day}',f'--cache_root={cache}',f'--cache_only={0 if gate else 1}','--replay_ns=32768','--write_output=0','--profile=1']
        out.append((f'cache_day{day}',cmd,'cache',day))
    def replay_task(name,day=1,segs=10,scale=1,npix=32768,rotation=0,limit=0):
        dest=root/'runs'/name
        cmd=[str(bin_dir/'replay'),'--gpus=3',f'--cache_root={cache}',f'--out_dir={dest}',f'--day={day}',f'--segs={segs}',f'--samples={npix}',f'--grid_scale={scale}',f'--rotation={rotation}',f'--segment_limit={limit}',f'--dcf_bin={dcf}']
        out.append((name,cmd,'replay',day))
    if phase=='pilot':cache_task(1,True);replay_task('pilot_day1_np4096_seg1',npix=4096,limit=1)
    if phase=='cache-rest':
        for day in (113,225,337,450):cache_task(day)
    if phase=='replay':
        # Predeclared cost reduction: fixed 8192 directions for five-day table,
        # fixed 4096 for all seven one-dimensional sweep points and frame test,
        # two independent 32768-direction density checks, retaining full sky V.
        for day in (1,113,225,337,450):replay_task(f'day{day}_s10_g1_n8192',day=day,npix=8192)
        for segs in (5,10,20,40):replay_task(f'day1_s{segs}_g1_n4096',segs=segs,npix=4096)
        for scale in (.75,1.25,1.5):replay_task(f'day1_s10_g{scale}_n4096',scale=scale,npix=4096)
        replay_task('day1_s10_g1_rot037_n4096',rotation=.37,npix=4096)
        for day in (1,450):replay_task(f'day{day}_s10_g1_n32768',day=day,npix=32768)
    return out

def run(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        files=[root/'bin/main_cache',root/'bin/replay',BASE/'earth_10Mhz/B_10M.bin',BASE/'system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin']
        files+=sorted((root/'source').glob('*'))
        identities={str(p):digest(p) for p in files if p.is_file()}
        freeze=root/'freeze.json'
        if freeze.exists() and json.loads(freeze.read_text())['sha256']!=identities:raise RuntimeError('frozen source/binary/input changed')
        if not freeze.exists():save(freeze,{'created_at':stamp(),'sha256':identities,'stage':'raw=host-reduced nm1; consumed=production float host phase correction to n','gpu':'physical3, G=1','timing':'shared-node correctness/progress only'})
        if args.phase=='replay':
            needed={day for name,_,_,day in tasks(root,args.phase) if not args.only or name==args.only}
            for day in sorted(needed):
                cached=json.loads((root/'checkpoints'/f'cache_day{day}.json').read_text())
                if cached.get('state')!='completed' or not all(Path(p).is_file() and digest(Path(p))==v for p,v in cached['artifacts'].items()):
                    raise RuntimeError('frozen cached production day failed verification: '+str(day))
        save(root/'status.json',{'state':'running','phase':args.phase,'at':stamp(),'pid':os.getpid()})
        for name,command,kind,day in tasks(root,args.phase):
            if args.only and name!=args.only:continue
            checkpoint=root/'checkpoints'/f'{name}.json'
            if checkpoint.exists():
                record=json.loads(checkpoint.read_text())
                if record['state']=='completed' and record['command']==command and all(Path(p).is_file() and digest(Path(p))==v for p,v in record['artifacts'].items()):continue
                if record['state']=='completed':raise RuntimeError('completed artifact verification failed: '+name)
                checkpoint.rename(checkpoint.with_name(checkpoint.name+'.interrupted.'+str(time.time_ns())))
            if (root/'PAUSE').exists():save(root/'status.json',{'state':'paused','next':name,'reason':'PAUSE file','at':stamp()});return 3
            if shutil.disk_usage(root).free<2*1024**3:raise RuntimeError('low disk guard <2GiB')
            try:snap=idle_snapshot()
            except GpuBusy as error:
                save(root/'status.json',{'state':'paused','next':name,'reason':str(error),'at':stamp()});return 3
            log=root/'logs'/f'{name}.log';log.parent.mkdir(exist_ok=True)
            if log.exists():log.rename(log.with_name(log.name+'.previous.'+str(time.time_ns())))
            state={'state':'running','command':command,'started_at':stamp(),'gpu_before':snap,'name':name}
            save(checkpoint,state);env=os.environ.copy();env['OMP_NUM_THREADS']='8';env.pop('CUDA_VISIBLE_DEVICES',None)
            start=time.monotonic()
            with log.open('wb') as stream:
                child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                try:rc=child.wait(timeout=args.timeout)
                except (BaseException,subprocess.TimeoutExpired):
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            text=log.read_text(errors='replace')
            if rc:raise RuntimeError(f'{name} returned {rc}; log={log}')
            if kind=='cache':
                files=list((root/'cache').glob(f'day{day}_seg*.bin'))
                if len(files)!=10 or text.count('REPLAY_CACHE ')!=10:raise RuntimeError('incomplete cached day '+name)
                if day==1:
                    files+=list((root/'cache').glob('production_gate_day1_*'))
                    if len(files)!=13 or 'PRODUCTION_SAMPLED_GATE ' not in text:raise RuntimeError('missing production sampled gate')
            else:
                if 'REPLAY_DONE ' not in text:raise RuntimeError('missing replay completion')
                files=[p for p in (root/'runs'/name).iterdir() if p.is_file()]
            artifacts={str(p):digest(p) for p in files+[log]}
            state.update(state='completed',finished_at=stamp(),returncode=rc,process_wall_s=time.monotonic()-start,artifacts=artifacts)
            save(checkpoint,state);print(json.dumps({'completed':name,'wall_s':state['process_wall_s'],'at':stamp()}),flush=True)
        save(root/'status.json',{'state':'completed','phase':args.phase,'at':stamp()});return 0

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    p.add_argument('--phase',choices=('pilot','cache-rest','replay'),required=True);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--only',help='run one exact named task from the phase')
    args=p.parse_args()
    def interrupted(signum,frame):raise KeyboardInterrupt('received signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGHUP,interrupted)
    try:return run(args)
    except BaseException as e:
        save(args.root/'status.json',{'state':'failed','phase':args.phase,'error':str(e),'at':stamp()})
        print(str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
