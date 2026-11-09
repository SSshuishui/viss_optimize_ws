#!/usr/bin/env python3
"""Matched CPU/GPU direct + actual-scale compact-ROI FINUFFT. No full-sky arrays."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import argparse,ctypes,hashlib,json,math,platform,resource,time,traceback
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def metric(v,r):
    d=v-r
    return dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(r)),max_absolute=float(np.max(np.abs(d))),
                rmse=float(np.sqrt(np.mean(np.abs(d)**2))))
def timing(v):
    return dict(n=len(v),individual_s=v,median_s=float(np.median(v)),min_s=float(np.min(v)),max_s=float(np.max(v)),
                sample_sd_s=float(np.std(v,ddof=1)),IQR_s=float(np.percentile(v,75)-np.percentile(v,25)))

p=argparse.ArgumentParser();p.add_argument('--geometry',type=Path,required=True);p.add_argument('--prefix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--cpu-only',action='store_true')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.time()
script=Path(__file__).resolve();libpath=script.with_name('direct.so');lib=ctypes.CDLL(str(libpath))
fn=lib.apply_direct
P=ctypes.c_void_p;I=ctypes.c_int
fn.argtypes=[I,I,P,P,P,P,I,I,I,P];fn.restype=None
def direct(s,t,c,M,sign=1,threads=1,out=None):
    if out is None:out=np.empty(len(t),dtype=np.complex128)
    fn(len(s),len(t),s.ctypes.data,t.ctypes.data,c.ctypes.data,M.ctypes.data if M is not None else 0,int(M is not None),sign,threads,out.ctypes.data)
    return out
z=np.load(a.geometry/'production_fp32_geometry_export_v1.npz')
uvw=np.ascontiguousarray(z['uvw_lambda_fp32'],dtype=np.float64)
px=np.ascontiguousarray(np.c_[z['lmn_fp32'][:,:2],z['nm1_fp32']],dtype=np.float64)
mask=np.ascontiguousarray(z['point_mask_u8'],dtype=np.uint8)
bp=np.ascontiguousarray(z['bp32'],dtype=np.complex128)
# The matched masked fixture uses every exported record, preserving exact frozen masks.
t0=time.perf_counter();y=direct(px,uvw,bp,np.ascontiguousarray(mask.T),sign=-1,threads=16);ys=time.perf_counter()-t0
prefixnames=['l.f32','m.f32','n.f32','nm1.f32','bp32.f32']
pre={n:np.fromfile(a.prefix/n,dtype='<f4',count=65536) for n in prefixnames}
assert all(len(v)==65536 for v in pre.values())
roi=np.ascontiguousarray(np.c_[pre['l.f32'],pre['m.f32'],pre['nm1.f32']],dtype=np.float64)
ub=np.ascontiguousarray(uvw[z['day']==1],dtype=np.float64)
rbp=np.ascontiguousarray(pre['bp32.f32'],dtype=np.complex128)
t0=time.perf_counter();ry=direct(roi,ub,rbp,None,sign=-1,threads=16);rys=time.perf_counter()-t0
datasets=[('frozen_fullsphere_masked_subset',uvw,px,y,mask),('actual_scale_compact_ROI_unmasked',ub,roi,ry,None)]
report=dict(schema='t05_matched_direct_roi_nufft_v1',status='running',repetitions=10,warmups=1,
    host=platform.node(),precision='CPU/GPU/FINUFFT complex128, FP64 phase/trig/accumulation from promoted frozen FP32 inputs',
    shared_node=True,performance_scope='controlled medium-size subproblem; host and other GPUs may have other users; no production speedup extrapolation',
    W='unit independent baseline weights, Hermitian factor2 for real dirty output',
    D='2*sum(M) for masked fixture; constant2*Nb for unmasked ROI; same across implementations within dataset',
    phase='matched artificial subproblem raw nm1 forward(-) and adjoint(+); no extra host correction',
    setup=dict(reduced_sky_masked_forward_s=ys,reduced_sky_ROI_unmasked_forward_s=rys),
    sources={str(a.geometry/'production_fp32_geometry_export_v1.npz'):sha(a.geometry/'production_fp32_geometry_export_v1.npz'),
      **{str(a.prefix/n):sha(a.prefix/n) for n in prefixnames},str(script):sha(script),str(script.with_name('direct.cpp')):sha(script.with_name('direct.cpp'))},
    limitations=['Masked fixture has5600 selected records over five days and2048 sampled sky pixels, not the full production image',
      'ROI has65536 contiguous original NEST pixels0..65535, intentionally restricted sky; no angular/UVW rescaling',
      'No-mask NUFFT is a Fourier subproblem, not a solution for baseline-dependent occultation',
      'Input visibilities are generated once from the same reduced sky and operator; not the full-sky64 visibility experiment',
      'Compared direct implementations are transparent FP64 kernels, not claimed identical to optimized production FP32 CUDA',
      'Setup/apply memory measurements are qualified; no continuous GPU-memory peak claim'],datasets={})
arrays={};cpu_refs={}
for name,s,t,c,M in datasets:
    d=2*(M.sum(axis=0).astype(np.uint64) if M is not None else np.full(len(t),len(s),dtype=np.uint64))
    t0=time.perf_counter();ref=direct(s,t,c,M,threads=1);refsetup=time.perf_counter()-t0
    cpu_refs[name]=ref; ds=dict(sources=len(s),targets=len(t),pairs=len(s)*len(t),mask_fraction=float(M.mean()) if M is not None else 1,
      coordinate_min_s=s.min(axis=0).tolist(),coordinate_max_s=s.max(axis=0).tolist(),
      coordinate_min_t=t.min(axis=0).tolist(),coordinate_max_t=t.max(axis=0).tolist(),
      y_sha256=hashlib.sha256(c.tobytes()).hexdigest(),first_reference_apply_s=refsetup,CPU={})
    for nt in (1,8,16):
        out=np.empty_like(ref);direct(s,t,c,M,threads=nt,out=out)
        times=[]
        for rep in range(10):
            t0=time.perf_counter();direct(s,t,c,M,threads=nt,out=out);times.append(time.perf_counter()-t0)
        ds['CPU'][str(nt)]=dict(apply=timing(times),error=metric(out,ref),bitwise_equal=bool(np.array_equal(out,ref)))
    report['datasets'][name]=ds
    arrays[name+'_reference']=ref;arrays[name+'_D']=d

if not a.cpu_only:
    import cupy as cp
    gpu_code=r'''
    extern "C" __global__ void apply(int ns,int nt,const double* s,const double* t,const double2* c,const unsigned char* M,int masked,double2* out){
      int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=nt)return;
      double ar=0,ai=0,x=t[3*p],y=t[3*p+1],z=t[3*p+2];
      for(int j=0;j<ns;j++){
        if(masked&&!M[(long long)j*nt+p])continue;
        double ph=6.283185307179586476925286766559*((s[3*j]*x+s[3*j+1]*y)+s[3*j+2]*z);
        double sn,cs;sincos(ph,&sn,&cs);double2 v=c[j];
        ar+=v.x*cs-v.y*sn;ai+=v.x*sn+v.y*cs;
      }out[p]=make_double2(ar,ai);
    }'''
    tg=time.perf_counter();kernel=cp.RawKernel(gpu_code,'apply',options=('--fmad=false',));kernel.compile();cp.cuda.Device().synchronize()
    report['GPU']=dict(physical_selection=os.getenv('CUDA_VISIBLE_DEVICES'),device_name=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
      compilation_context_s=time.perf_counter()-tg)
    for name,s,t,c,M in datasets:
        tg=time.perf_counter();gs,gt,gc=cp.asarray(s),cp.asarray(t),cp.asarray(c);gm=cp.asarray(M) if M is not None else cp.zeros(1,dtype=cp.uint8);go=cp.empty(len(t),dtype=cp.complex128)
        cp.cuda.Device().synchronize();setup=time.perf_counter()-tg
        args=(np.int32(len(s)),np.int32(len(t)),gs,gt,gc,gm,np.int32(M is not None),go)
        kernel(((len(t)+127)//128,),(128,),args);cp.cuda.Device().synchronize()
        wall=[];evt=[]
        for rep in range(10):
            e1,e2=cp.cuda.Event(),cp.cuda.Event();tg=time.perf_counter();e1.record();kernel(((len(t)+127)//128,),(128,),args);e2.record();e2.synchronize();wall.append(time.perf_counter()-tg);evt.append(cp.cuda.get_elapsed_time(e1,e2)*.001)
        tg=time.perf_counter();got=cp.asnumpy(go);d2h=time.perf_counter()-tg
        report['datasets'][name]['GPU']=dict(setup_H2D_s=setup,apply_wall=timing(wall),apply_kernel_events=timing(evt),D2H_s=d2h,
           error=metric(got,cpu_refs[name]),explicit_device_array_bytes=int(sum(q.nbytes for q in [gs,gt,gc,gm,go])),
           pool_used_bytes=int(cp.get_default_memory_pool().used_bytes()),pool_reserved_bytes=int(cp.get_default_memory_pool().total_bytes()),
           memory_method='explicit arrays and postapply CuPy pool accounting; excludes context/driver/transient compiler, not sampled peak')
        arrays[name+'_gpu']=got
        del gs,gt,gc,gm,go;cp.get_default_memory_pool().free_all_blocks()

import finufft
try:
    # Actual baseline lambda coordinates as sources; actual frozen nm1 sky ROI as targets.
    s=np.ascontiguousarray(ub*2*np.pi); t=roi
    nufft={}; outputs={}; ref=cpu_refs['actual_scale_compact_ROI_unmasked']
    for eps in (1e-6,1e-9):
        t0=time.perf_counter();plan=finufft.Plan(3,3,n_trans=1,eps=eps,isign=1,dtype='complex128',nthreads=16)
        plan.setpts(x=np.ascontiguousarray(s[:,0]),y=np.ascontiguousarray(s[:,1]),z=np.ascontiguousarray(s[:,2]),
           s=np.ascontiguousarray(t[:,0]),t=np.ascontiguousarray(t[:,1]),u=np.ascontiguousarray(t[:,2]));setup=time.perf_counter()-t0
        out=plan.execute(ry);times=[]
        for rep in range(10):
            t0=time.perf_counter();out=plan.execute(ry);times.append(time.perf_counter()-t0)
        nufft[str(eps)]=dict(setup_plan_setpts_s=setup,apply=timing(times),error=metric(out,ref),
            max_RSS_MiB=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024),
            memory_method='process lifetime high-water RSS at this point; includes earlier CPU/GPU work, not isolated NUFFT allocation')
        arrays['ROI_nufft_eps_'+str(eps)]=out
    report['FINUFFT']=dict(version=finufft.__version__,operator='type3 positive sign exp(+i*(2pi*uvw).nm1) at original coordinates; no divide by100',
        scope='actual_scale_compact_ROI_unmasked only',nthreads=16,results=nufft)
except Exception as exc:
    report['FINUFFT']=dict(status='failed',error=repr(exc),traceback=traceback.format_exc())
report['status']='completed' if 'results' in report.get('FINUFFT',{}) else 'completed_with_NUFFT_failure'
report['elapsed_s']=time.time()-start;report['peak_process_RSS_MiB']=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
np.savez_compressed(a.output/'outputs.npz',**arrays)
(a.output/'matched.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
for name,v in report['datasets'].items():
    print(name,{k:v['CPU'][k]['apply']['median_s'] for k in v['CPU']},v.get('GPU',{}).get('error'))
print('FINUFFT',json.dumps(report['FINUFFT']));print('status',report['status'],'elapsed_s',report['elapsed_s'])
