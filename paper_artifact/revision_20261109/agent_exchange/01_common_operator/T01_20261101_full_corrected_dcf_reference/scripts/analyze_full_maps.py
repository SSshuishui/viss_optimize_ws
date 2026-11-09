#!/usr/bin/env python3
"""Stream all three temporary C maps; no full-map copy or coordinate inference."""
import argparse,datetime,hashlib,json,os
from pathlib import Path
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import numpy as np

class Moments:
    def __init__(self):
        self.n=0;self.mx=np.longdouble(0);self.my=np.longdouble(0);self.xx=np.longdouble(0);self.yy=np.longdouble(0);self.xy=np.longdouble(0);self.err=np.longdouble(0);self.lo=float('inf');self.hi=-float('inf');self.maxerr=0.
    def add(self,x,y):
        n=len(x)
        if not n:return
        mx=np.mean(x,dtype=np.longdouble);my=np.mean(y,dtype=np.longdouble);dx=x-float(mx);dy=y-float(my)
        xx=np.sum(dx*dx,dtype=np.longdouble);yy=np.sum(dy*dy,dtype=np.longdouble);xy=np.sum(dx*dy,dtype=np.longdouble)
        if self.n:
            nx=self.n+n;cx=mx-self.mx;cy=my-self.my;k=np.longdouble(self.n)*n/nx
            self.xx+=xx+cx*cx*k;self.yy+=yy+cy*cy*k;self.xy+=xy+cx*cy*k;self.mx+=cx*n/nx;self.my+=cy*n/nx;self.n=nx
        else:self.n=n;self.mx=mx;self.my=my;self.xx=xx;self.yy=yy;self.xy=xy
        z=y-x;self.err+=np.sum(z*z,dtype=np.longdouble);self.maxerr=max(self.maxerr,float(np.max(np.abs(z))));self.lo=min(self.lo,float(x.min()));self.hi=max(self.hi,float(x.max()))
    def result(self,affine=None,global_std=None):
        if not self.n:return {'count':0}
        std=float(np.sqrt(self.xx/self.n));norm=float(self.xx+self.n*self.mx*self.mx);rmse=float(np.sqrt(self.err/self.n))
        r={'count':self.n,'reference_mean':float(self.mx),'candidate_mean':float(self.my),'reference_std':std,'relative_l2':float(np.sqrt(self.err/norm)) if norm>0 else None,'RMSE':rmse,'RMSE_over_reference_std':rmse/std if std>0 else None,'max_abs_error':self.maxerr,'pearson':float(self.xy/np.sqrt(self.xx*self.yy)) if self.xx>0 and self.yy>0 else None,'reference_min':self.lo,'reference_max':self.hi}
        if global_std is not None:r['RMSE_over_full_reference_std']=rmse/global_std if global_std>0 else None
        if affine is not None:
            a,b=map(np.longdouble,affine);mu=a*self.my+b-self.mx;sse=max(np.longdouble(0),a*a*self.yy+self.xx-2*a*self.xy+self.n*mu*mu);v=float(np.sqrt(sse/self.n))
            r['same_global_affine_after']={'alpha':float(a),'beta':float(b),'RMSE':v,'RMSE_over_reference_std':v/std if std>0 else None,'relative_l2':float(np.sqrt(sse/norm)) if norm>0 else None}
        return r
    def affine(self):
        a=self.xy/self.yy if self.yy>0 else np.longdouble(0);return a,self.mx-a*self.my

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);a=ap.parse_args();root=a.root.resolve();npix=201326592;nparent=npix//1024;CH=1<<20
    folders=[root/'runs/reference',root/'runs/sts_grid',root/'runs/sts_actual']
    paths=[folders[0]/'corrected_dcf_C.f32',folders[1]/'sts_grid_C.f32',folders[2]/'sts_actual_C.f32',folders[0]/'native_regions.u8']
    sizes=[npix*4]*3+[npix]
    for p,size in zip(paths,sizes):assert p.stat().st_size==size,(str(p),p.stat().st_size,size)
    meta=[json.loads((f/'full_stats.json').read_text()) for f in folders]
    expected=[m['C']['sha256'] for m in meta]+[meta[0]['native_regions']['sha256']]
    hashes=[hashlib.sha256() for _ in paths];handles=[p.open('rb') for p in paths]
    moments=[[Moments() for _ in range(4)] for _ in range(2)];population=np.zeros(4,dtype='i8')
    parent={name:np.zeros(nparent,dtype='f8') for name in ('ref_mean','grid_mean','actual_mean','grid_residual_mean','actual_residual_mean','grid_residual_rms','actual_residual_rms')}
    parent['valid_count']=np.zeros(nparent,dtype='u2');parent['region_count']=np.zeros((nparent,3),dtype='u2');parent['parent_ids']=np.arange(nparent,dtype='u4')
    try:
        for off in range(0,npix,CH):
            take=min(CH,npix-off);raw=[h.read(take*(4 if i<3 else 1)) for i,h in enumerate(handles)]
            for i,b in enumerate(raw):assert len(b)==take*(4 if i<3 else 1);hashes[i].update(b)
            maps=[np.frombuffer(raw[i],dtype='<f4').astype('f8') for i in range(3)];tags=np.frombuffer(raw[3],dtype='u1');assert np.isin(tags,[0,1,2,255]).all()
            for v in maps:assert np.isfinite(v).all()
            valid=tags!=255;population[:3]+=[np.count_nonzero(tags==r) for r in range(3)];population[3]+=np.count_nonzero(~valid)
            x=maps[0]
            for j,y in enumerate(maps[1:]):
                moments[j][0].add(x[valid],y[valid])
                for r in range(3):sel=tags==r;moments[j][r+1].add(x[sel],y[sel])
            parents=slice(off//1024,(off+take)//1024);v=valid.reshape(-1,1024);count=v.sum(axis=1);parent['valid_count'][parents]=count
            for r in range(3):parent['region_count'][parents,r]=(tags.reshape(-1,1024)==r).sum(axis=1)
            for label,values in zip(('ref','grid','actual'),maps):
                s=np.where(v,values.reshape(-1,1024),0).sum(axis=1,dtype='f8');parent[label+'_mean'][parents]=np.divide(s,count,out=np.zeros_like(s),where=count>0)
            for label,y in zip(('grid','actual'),maps[1:]):
                d=(y-x).reshape(-1,1024);mean=np.where(v,d,0).sum(axis=1,dtype='f8');square=np.where(v,d*d,0).sum(axis=1,dtype='f8')
                parent[label+'_residual_mean'][parents]=np.divide(mean,count,out=np.zeros_like(mean),where=count>0);parent[label+'_residual_rms'][parents]=np.sqrt(np.divide(square,count,out=np.zeros_like(square),where=count>0))
            if off%(CH*16)==0:print(f'FULL_STATS_CHILDREN {off+take}/{npix}',flush=True)
        for h in handles:assert h.read(1)==b''
    finally:
        for h in handles:h.close()
    for p,h,s in zip(paths,hashes,expected):assert h.hexdigest()==s,('SHA mismatch',str(p),h.hexdigest(),s)
    assert population.sum()==npix and population.tolist()==meta[0]['native_regions']['counts']
    assert population[3]==meta[0]['W']['zero_count']
    result={};qa={};names=['full_reference_valid','native_abs_lat_le30','native_abs_lat_30to60','native_abs_lat_gt60']
    for j,label in enumerate(('original_grid','actual_uv_candidate')):
        global_m=moments[j][0];affine=global_m.affine();gstd=float(np.sqrt(global_m.xx/global_m.n))
        result[label]={'global_affine_candidate_to_reference':{'alpha':float(affine[0]),'beta':float(affine[1]),'fitting_domain':'all reference-valid full-resolution pixels'},'domains':{name:m.result(affine,gstd) for name,m in zip(names,moments[j])}}
        p_label=('grid','actual')[j];recovered=np.sqrt(np.sum(parent[p_label+'_residual_rms']**2*parent['valid_count'],dtype='f8')/parent['valid_count'].sum());measured=global_m.result()['RMSE'];rel=float(abs(recovered-measured)/max(measured,1e-300));assert rel<=1e-10
        qa[label]={'parent_weighted_RMS':float(recovered),'full_resolution_RMSE':measured,'relative_reconstruction_difference':rel}
    out=root/'metrics';out.mkdir(exist_ok=True);np.savez_compressed(out/'parent_nside128.npz',**parent)
    parent_path=out/'parent_nside128.npz'
    artifact={'state':'completed','at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'pixels':npix,'valid_reference_pixels':int(npix-population[3]),'native_region_counts_including_W0':population.tolist(),'region_definition':'actual GPU n and referenceW, not Galactic; 0 abs(n)<=.5;1<=sqrt(3)/2;2higher;255W0','metrics':result,'parent_QA':qa,'temporary_artifacts':[{'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest(),'state':'validated_pending_cleanup'} for p,h in zip(paths,hashes)],'parent_artifact':'metrics/parent_nside128.npz','parent_artifact_bytes':parent_path.stat().st_size,'parent_artifact_sha256':hashlib.sha256(parent_path.read_bytes()).hexdigest(),'scope':'full201M day1 correctedDCF diagnosticN/W versus sameV STS; uncalibrated metrics primary, single global affine separately; no forward/450day/sky recovery claim'}
    (root/'full_domain_metrics.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'state':'completed','pixels':npix,'valid':artifact['valid_reference_pixels'],'metrics':result,'parent_QA':qa},indent=2))
if __name__=='__main__':main()
