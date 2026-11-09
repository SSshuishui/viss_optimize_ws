#!/usr/bin/env python3
"""CPU-only postprocessing of completed direct/NUFFT output arrays."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]); j=json.loads((p/'matched.json').read_text());a=np.load(p/'outputs.npz');out={}
def stats(v,r,D):
    m=D>0;x=2*v.real[m]/D[m];y=2*r.real[m]/D[m];d=x-y
    return dict(pixels=int(m.sum()),relative_l2=float(np.linalg.norm(d)/np.linalg.norm(y)),
      absolute_rmse=float(np.sqrt(np.mean(d*d))),std_normalized_rmse=float(np.sqrt(np.mean(d*d))/np.std(y)),
      max_absolute=float(np.max(np.abs(d))),affine_fit=False)
for name,v in j['datasets'].items():
    D=a[name+'_D'];ref=a[name+'_reference'];r={}
    if name+'_gpu' in a:r['GPU_vs_CPU']=stats(a[name+'_gpu'],ref,D)
    if 'ROI_unmasked' in name:
        for eps in ('1e-06','1e-09'):
            key='ROI_nufft_eps_'+eps
            if key in a:r['FINUFFT_'+eps+'_vs_CPU']=stats(a[key],ref,D)
    out[name]=r
result=dict(status='completed',normalization='2*real(complex response)/D, using identical D for candidate and CPU reference; D=0 excluded',
    source_matched_sha256=hashlib.sha256((p/'matched.json').read_bytes()).hexdigest(),
    source_outputs_sha256=hashlib.sha256((p/'outputs.npz').read_bytes()).hexdigest(),results=out)
(p/'dirty_metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2))
