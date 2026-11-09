#!/usr/bin/env python3
import fcntl,json,sys
from pathlib import Path
import numpy as np
import run_short_gate as core

def main():
    root=Path(__file__).resolve().parent.parent
    with (root/'execution.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        core.save(root/'combined_protocol.json',{'at':core.now(),'repeats':3,'gate':{'rawN_relative_l2':1e-3,'rawN_max_abs_over_peak':1e-3},'segments':1,'half_baselines':195468,'output_directions':8192,'same_cached_consumed_n_V':True,'dispatch':'original 512 baseline chunks, explicit mixed task IDs, atomic candidate','full_CLI':'compiled only, not executed','formal_timing':False,'new_forward':False})
        results={}
        for j in range(1,4):
            name=f'seg1_parallel_combined_r{j}';d=core.case(root,name,'replay_legacy_fastpath_combined','tasklist_parallel',compute_w=False)
            against={}
            for mode in (0,1):
                a=np.fromfile(root/'runs'/name/f'raw_N_mode{mode}.f32',dtype='<f4');b=np.fromfile(root/'runs/seg1_direct_fixed'/f'raw_N_mode{mode}.f32',dtype='<f4')
                against[str(mode)]=core.compare(a,b)
            results[name]={'CPU_gate_pass':d['gate_pass'],'all8192_vs_fixed_direct':against}
        passed=all(d['CPU_gate_pass'] and all(x['relative_l2']<=1e-3 and x['max_abs_over_peak']<=1e-3 for x in d['all8192_vs_fixed_direct'].values()) for d in results.values())
        report={'state':'completed','at':core.now(),'all_three_pass':passed,'results':results,'scope':'short firstsegment/8192 output only; no production full-map, forward or performance validation'}
        core.save(root/'metrics/combined_gate.json',report)
        summary=json.loads((root/'summary.json').read_text());summary['combined_candidate']=report;core.save(root/'summary.json',summary)
        core.save(root/'status.json',{'state':'completed','at':core.now(),'cases':14,'combined_three_pass':passed,'full_CLI':'compiled only, not executed','GPU_released':True})
        print(json.dumps(report),flush=True)
        if not passed:raise RuntimeError('combined gate failed; all outputs preserved')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        core.save(Path(__file__).resolve().parent.parent/'status.json',{'state':'prepared_or_failed','at':core.now(),'error':str(e),'no_wait_for_busy_GPU':True});raise
