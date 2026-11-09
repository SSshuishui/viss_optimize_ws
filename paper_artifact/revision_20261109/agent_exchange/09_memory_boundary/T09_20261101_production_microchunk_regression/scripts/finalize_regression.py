#!/usr/bin/env python3
"""Budgeted CPU independent inverse and complete-field equality gates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def error(a,b):
 a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);assert np.isfinite(a).all() and np.isfinite(b).all()
 return dict(relative_l2=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300)),max_abs_over_peak=float(np.max(abs(a-b))/max(np.max(abs(b)),1e-300)))
def main():
 names=('m0_default','m4_default','m4_small');checks={};oracles={}
 for name in names:
  p=ROOT/'runs'/name;subprocess.run([sys.executable,str(ROOT/'scripts/cpu_inverse_oracle.py'),'--root='+str(ROOT),'--case='+name],check=True)
  o=load(p/'CPU_INVERSE_ORACLE.json');oracles[name]=o;checks['CPU_'+name]=o['state']=='passed' and o['canonical_counts_exact_nonempty'] and o['raw_N_relative_l2_vs_FP64_phase']<=1e-3
 for field in ('N','W','C'):
  a=load(ROOT/'runs/m4_default'/f'full_{field}.json');b=load(ROOT/'runs/m4_small'/f'full_{field}.json');checks['MODE4_full_'+field]=a['sha256']==b['sha256'] and a['pixels']==b['pixels']==3221225472
 for n in ('forward_raw.f32x2','forward_consumed.f32x2','sample_ids.u32','sample_N.f32','sample_W.u32','sample_C.f32'):checks['MODE4_bitwise_'+n]=sha(ROOT/'runs/m4_default'/n)==sha(ROOT/'runs/m4_small'/n)
 checks['all_mode_fullW']=len({load(ROOT/'runs'/n/'full_W.json')['sha256'] for n in names})==1
 ids=((2*np.arange(2048,dtype=np.uint64)+1)*3221225472//4096).astype('<u4')
 for n in names:checks['sample_ids_'+n]=np.array_equal(np.fromfile(ROOT/'runs'/n/'sample_ids.u32','<u4'),ids)
 stats={n:{f:load(ROOT/'runs'/n/f'full_{f}.json') for f in ('N','W','C')} for n in names}
 sample={f:error(np.fromfile(ROOT/'runs/m4_default'/f'sample_{f}.f32','<f4'),np.fromfile(ROOT/'runs/m0_default'/f'sample_{f}.f32','<f4')) for f in ('N','C')}
 result={'state':'passed' if all(checks.values()) else 'failed','gate':{'pass':all(checks.values())},'checks':checks,'independent_inverse':{n:{k:v for k,v in o.items() if k not in ('probes','GPU_N','CPU_N_FP64_phase')} for n,o in oracles.items()},'MODE4_vs_sameG2_original_MODE0_sample_differences':sample,'full_stats':stats,'scope':'actual full production first original segment at G2, complete half forward and complete output domain; sampled errors are descriptive, full IO stability gated by SHA; not performance timing or physical exactness'}
 (ROOT/'metrics/FINAL_QA.json').write_text(json.dumps(result,indent=2)+'\n');assert all(checks.values()),'preserved numerical gate failure; no tolerance adjustment'
if __name__=='__main__':main()
