#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--reference', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--chunk', type=int, default=4_000_000)
    p.add_argument('--json-out', type=Path)
    return p.parse_args()

def main():
    a=args()
    if a.reference.stat().st_size != a.candidate.stat().st_size:
        raise SystemExit('map file sizes differ')
    n=a.reference.stat().st_size//4
    x=np.memmap(a.reference,dtype=np.float32,mode='r',shape=(n,))
    y=np.memmap(a.candidate,dtype=np.float32,mode='r',shape=(n,))
    sx=sy=sxx=syy=sxy=0.0; xmin=float('inf'); xmax=float('-inf'); valid=0
    for i in range(0,n,a.chunk):
        xx=np.asarray(x[i:i+a.chunk],dtype=np.float64); yy=np.asarray(y[i:i+a.chunk],dtype=np.float64)
        ok=np.isfinite(xx)&np.isfinite(yy); xx=xx[ok]; yy=yy[ok]
        if not len(xx): continue
        valid+=len(xx); sx+=xx.sum(); sy+=yy.sum(); sxx+=np.dot(xx,xx); syy+=np.dot(yy,yy); sxy+=np.dot(xx,yy)
        xmin=min(xmin,float(xx.min())); xmax=max(xmax,float(xx.max()))
    mx=sx/valid; my=sy/valid; varx=sxx-valid*mx*mx; vary=syy-valid*my*my; cov=sxy-valid*mx*my
    alpha=cov/vary if vary>0 else 0.0; beta=mx-alpha*my
    pearson=cov/np.sqrt(varx*vary) if varx>0 and vary>0 else 0.0
    raw=aff=ref=0.0
    for i in range(0,n,a.chunk):
        xx=np.asarray(x[i:i+a.chunk],dtype=np.float64); yy=np.asarray(y[i:i+a.chunk],dtype=np.float64)
        ok=np.isfinite(xx)&np.isfinite(yy); xx=xx[ok]; yy=yy[ok]
        raw+=np.square(xx-yy).sum(); aff+=np.square(xx-(alpha*yy+beta)).sum(); ref+=np.square(xx).sum()
    dr=xmax-xmin
    out={'n':int(valid),'pearson':float(pearson),'affine_gain':float(alpha),'affine_bias':float(beta),
         'raw_rmse':float(np.sqrt(raw/valid)),
         'raw_nrmse_range':float(np.sqrt(raw/valid)/dr) if dr else None,
         'affine_nrmse_range':float(np.sqrt(aff/valid)/dr) if dr else None,
         'relative_l2':float(np.sqrt(raw/ref)) if ref else None}
    print(json.dumps(out,indent=2))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True,exist_ok=True)
        a.json_out.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__': main()
