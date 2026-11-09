#!/usr/bin/env python3
"""Two short actual-representative LS diagnostics, without modifying frozen replay."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from run_batch import idle_snapshot,save,stamp,digest,GpuBusy

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);args=p.parse_args();root=args.root
    with (root/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        case=root/'runs/day1_s10_g1_n4096';binary=root/'solver_bin/one_step'
        if not (case/'metrics.json').exists() or not json.loads((case/'metrics.json').read_text())['oracle']['pass']:raise RuntimeError('required main reference oracle is not complete')
        results=[];identity={str(p):digest(p) for p in [binary,root/'scripts/one_step.cu',root/'scripts/run_one_step.py',root/'source/replay.cu',root/'freeze.json',case/'ids.u32',case/'l.f32',case/'m.f32',case/'n.f32',case/'direct_N.f64',case/'rep_N.f32',case/'rep_W.u32']}
        for noise in (False,True):
            label='noise20dB_seed101' if noise else 'noiseless';dest=(root/'solver_noise_seed101/one_step.json') if noise else root/'solver/noiseless.json';dest.parent.mkdir(parents=True,exist_ok=True)
            snap=idle_snapshot();command=[str(binary),'--gpus=3',f'--cache_root={root}/cache',f'--case_dir={case}',f'--output={dest}','--day=1']
            if noise:command += ['--noise_seed=101',f'--noise_root={root}/solver_noise_seed101','--dcf_bin=/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin']
            log=root/'logs'/('one_step_'+label+'.log');execution={'label':label,'command':command,'state':'running','at':stamp(),'gpu_before':snap}
            save(root/'solver'/f'{label}_execution.json',execution)
            with log.open('wb') as out:
                child=subprocess.Popen(command,stdout=out,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'8'})
                try:rc=child.wait(timeout=180)
                except BaseException:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            if rc:raise RuntimeError(label+' failed; see '+str(log))
            data=json.loads(dest.read_text())
            if not data['adjoint_scalar_controls_pass']:raise RuntimeError('actual scalar adjoint controls failed')
            data['positive_rep_direction_is_descent']=data['g_dot_d']>0
            data['rep_optimal_alpha_nonnegative']=data['optimal_line_alpha_d']>=0
            data['final_loss_true_direction']=data['initial_loss']-data['line_loss_decrease_g']
            data['final_loss_rep_direction']=data['initial_loss']-data['line_loss_decrease_d']
            save(dest,data)
            execution.update(state='completed',finished_at=stamp(),returncode=0,output_sha256=digest(dest),log_sha256=digest(log));save(root/'solver'/f'{label}_execution.json',execution);results.append(data)
            print(json.dumps({'done':label,'cosine':data['direction_cosine'],'loss_decrease_ratio':data['decrease_d_over_g'],'gpu_wall_s':data['kernel_plus_D2H_wall_s']}),flush=True)
        save(root/'solver/summary.json',{'created_at':stamp(),'results':results,'identity_sha256':identity,'scope':'same restricted real4096-direction LS objective, cached fullsky visibility, actual original representative kernel; exact line search in each direction; one step is not convergence'})
        print('SOLVER_GPU_COMPLETE',flush=True)
        # Independent CPU oracle also verifies the noisy actual-replay result.
        subprocess.run([sys.executable,str(root/'scripts/analyze.py'),'--root',str(root/'solver_noise_seed101')],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        subprocess.run([sys.executable,str(root/'scripts/full_cpu_oracle.py'),'--root',str(root/'solver_noise_seed101')],check=True,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        temporary=[]
        for path in sorted((root/'solver_noise_seed101/cache').glob('day1_seg*.bin')):
            temporary.append({'path':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'retained':False});path.unlink()
        save(root/'solver/noise_temporary_cache_manifest.json',{'removed_only_after_noise_CPU_oracle_pass':temporary,'recreate':'run the recorded solver command against frozen original cache, source, binary and seed101'})
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except GpuBusy as error:print('PAUSED '+str(error),file=sys.stderr);sys.exit(3)
