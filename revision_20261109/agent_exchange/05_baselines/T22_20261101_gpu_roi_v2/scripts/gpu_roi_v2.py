#!/usr/bin/env python3
"""Same-process FP64 GPU direct vs cuFINUFFT, actual-scale frozen T05 ROI."""
import os
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import ctypes,gc,hashlib,json,time,traceback,uuid
from pathlib import Path
import numpy as np
R=Path(os.environ['PARCO_REVISION_DATA_ROOT']).resolve();P=Path(__file__).resolve().parents[1];O=P/'metrics';O.mkdir(exist_ok=True)
T=R/'agent_exchange/05_baselines/T05_20261101_matched';G=R/'geometry';F=R/'precision/queued_correctness_v2/pilot8_dump_v2/prefix_dump'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metric(v,r):return dict(relative_l2=float(np.linalg.norm(v-r)/np.linalg.norm(r)),max_absolute=float(np.max(np.abs(v-r))))
def times(v):return dict(n=len(v),individual_s=v,median_s=float(np.median(v)),sample_sd_s=float(np.std(v,ddof=1)),IQR_s=float(np.percentile(v,75)-np.percentile(v,25)))
out=dict(status='running',GPU_physical=os.getenv('CUDA_VISIBLE_DEVICES'),start=time.time(),warmups=1,repeats=10,shared_node=True,
    scope='1120 actual day1 half-baselines x original NEST prefix65536 pixels, no mask, no DCF; actual UVW scale, no divide-by100',
    phase='matched raw nm1 Fourier subproblem; positive adjoint, not production host-correction replay')
try:
    z=np.load(G/'production_fp32_geometry_export_v1.npz');ub=np.ascontiguousarray(z['uvw_lambda_fp32'][z['day']==1],dtype=np.float64)
    # Explicit stacking avoids any transformed coordinate or sky-scale change.
    roi=np.ascontiguousarray(np.column_stack([np.fromfile(F/n,dtype='<f4',count=65536) for n in ('l.f32','m.f32','nm1.f32')]),dtype=np.float64)
    bp=np.ascontiguousarray(np.fromfile(F/'bp32.f32',dtype='<f4',count=65536),dtype=np.complex128)
    lib=ctypes.CDLL(str(T/'scripts/direct.so'));fn=lib.apply_direct;C=ctypes.c_int;V=ctypes.c_void_p;fn.argtypes=[C,C,V,V,V,V,C,C,C,V]
    y=np.empty(len(ub),np.complex128);fn(len(roi),len(ub),roi.ctypes.data,ub.ctypes.data,bp.ctypes.data,0,0,-1,16,y.ctypes.data)
    old=json.loads((T/'metrics/matched.json').read_text());name='actual_scale_compact_ROI_unmasked'
    assert hashlib.sha256(y.tobytes()).hexdigest()==old['datasets'][name]['y_sha256'],'V differs from frozen T05'
    with np.load(T/'metrics/outputs.npz') as q:ref=q[name+'_reference'].copy();direct=q[name+'_gpu'].copy();D=q[name+'_D'].copy()
    import cupy as cp,cufinufft
    cp.cuda.Device().synchronize();gstart=time.time();out['version']=cufinufft.__version__
    # Runtime identity only after the authorized wait/preflight and normal context init.
    class CUuuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
    driver=ctypes.CDLL('libcuda.so.1');dev=ctypes.c_int();driver.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    if driver.cuDeviceGet(ctypes.byref(dev),0)!=0:raise RuntimeError('cuDeviceGet identity lookup failed')
    uid=CUuuid();uuidfn=getattr(driver,'cuDeviceGetUuid_v2',driver.cuDeviceGetUuid);uuidfn.argtypes=[ctypes.POINTER(CUuuid),ctypes.c_int]
    if uuidfn(ctypes.byref(uid),dev.value)!=0:raise RuntimeError('cuDeviceGetUuid identity lookup failed')
    actual_uuid='GPU-'+str(uuid.UUID(bytes=bytes(uid.bytes)))
    out['actual_device_identity']=dict(logical_device=0,CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'),CUDA_DEVICE_ORDER=os.getenv('CUDA_DEVICE_ORDER'),
       pci_bus_id=cp.cuda.runtime.deviceGetPCIBusId(0),uuid=actual_uuid)
    chain=P/'chain_state.json'
    if chain.exists():
        pf=json.loads(chain.read_text()).get('one_time_preflight')
        if pf:
            out['actual_device_identity'].update(preflight_uuid=pf['uuid'],preflight_uuid_matches=actual_uuid==pf['uuid'])
            if actual_uuid!=pf['uuid']:raise RuntimeError('actual CUDA device UUID does not match guarded physical GPU2')
    sx=np.array([.2,.4]);sy=np.zeros(2);cc=np.array([1+2j,.3-.7j]);tt=np.array([.5,1.2]);cpu=np.exp(-1j*tt[:,None]*sx)@cc
    gx,zero,gc_,gt=cp.asarray(sx),cp.asarray(sy),cp.asarray(cc),cp.asarray(tt)
    check=cp.asnumpy(cufinufft.nufft3d3(gx,zero,zero,gc_,gt,zero,zero,eps=1e-12,isign=-1))
    out['tiny_isign_minus1_check']=metric(check,cpu);assert out['tiny_isign_minus1_check']['relative_l2']<1e-10
    del gx,zero,gc_,gt;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    t0=time.perf_counter();s=[cp.asarray(np.ascontiguousarray(2*np.pi*ub[:,k])) for k in range(3)];t=[cp.asarray(np.ascontiguousarray(roi[:,k])) for k in range(3)];gy=cp.asarray(y);go=cp.empty(len(roi),cp.complex128);cp.cuda.Device().synchronize();out['H2D_s']=time.perf_counter()-t0
    out['explicit_device_array_bytes']=int(sum(v.nbytes for v in [*s,*t,gy,go]));baseline=cp.cuda.runtime.memGetInfo();out['memory_before_plan_free_total']=list(map(int,baseline));results={}
    arrays={}
    code=r'''
    extern "C" __global__ void direct_apply(int ns,int nt,const double* s,const double* t,const double2* c,double2* out){
      int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=nt)return;
      double ar=0,ai=0,x=t[3*p],y=t[3*p+1],z=t[3*p+2];
      for(int j=0;j<ns;j++){
        double ph=6.283185307179586476925286766559*((s[3*j]*x+s[3*j+1]*y)+s[3*j+2]*z);
        double sn,cs;sincos(ph,&sn,&cs);double2 v=c[j];
        ar+=v.x*cs-v.y*sn;ai+=v.x*sn+v.y*cs;
      }out[p]=make_double2(ar,ai);
    }'''
    t0=time.perf_counter();kernel=cp.RawKernel(code,'direct_apply',options=('--fmad=false',));kernel.compile();cp.cuda.Device().synchronize();compile_s=time.perf_counter()-t0
    t0=time.perf_counter();ds,dt=cp.asarray(ub),cp.asarray(roi);dgo=cp.empty(len(roi),cp.complex128);cp.cuda.Device().synchronize();dsetup=time.perf_counter()-t0
    args=(np.int32(len(ub)),np.int32(len(roi)),ds,dt,gy,dgo)
    kernel(((len(roi)+127)//128,),(128,),args);cp.cuda.Device().synchronize();dv=[];de=[]
    for rep in range(10):
        e1,e2=cp.cuda.Event(),cp.cuda.Event();t0=time.perf_counter();e1.record();kernel(((len(roi)+127)//128,),(128,),args);e2.record();e2.synchronize();dv.append(time.perf_counter()-t0);de.append(cp.cuda.get_elapsed_time(e1,e2)*.001)
    t0=time.perf_counter();direct=cp.asnumpy(dgo);dd2h=time.perf_counter()-t0
    out['GPU_direct_sameprocess']=dict(compile_s=compile_s,setup_H2D_s=dsetup,apply_wall=times(dv),apply_events=times(de),D2H_s=dd2h,
       error_vs_CPU=metric(direct,ref),normalized_dirty_error=metric(2*direct.real/D,2*ref.real/D),
       explicit_device_array_bytes=int(ds.nbytes+dt.nbytes+gy.nbytes+dgo.nbytes),phase='positive2pi*(u*l+v*m+w*nm1); same frozen FP64 input V as cuFINUFFT')
    arrays['sameprocess_direct']=direct
    del ds,dt,dgo,args;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    for eps in (1e-6,1e-9):
        t0=time.perf_counter();plan=cufinufft.Plan(3,3,n_trans=1,eps=eps,isign=1,dtype='complex128');plan.setpts(*s,*t);cp.cuda.Device().synchronize();setup=time.perf_counter()-t0
        after_setup=cp.cuda.runtime.memGetInfo();plan.execute(gy,out=go);cp.cuda.Device().synchronize();vals=[];evt=[]
        for rep in range(10):
            e1,e2=cp.cuda.Event(),cp.cuda.Event();t0=time.perf_counter();e1.record();plan.execute(gy,out=go);e2.record();e2.synchronize();vals.append(time.perf_counter()-t0);evt.append(cp.cuda.get_elapsed_time(e1,e2)*.001)
        after_apply=cp.cuda.runtime.memGetInfo();t0=time.perf_counter();got=cp.asnumpy(go);d2h=time.perf_counter()-t0
        dirty=2*got.real/D;refdirty=2*ref.real/D
        results[str(eps)]=dict(setup_plan_setpts_s=setup,apply_wall=times(vals),apply_events=times(evt),D2H_s=d2h,error_vs_CPU=metric(got,ref),error_vs_GPU_direct=metric(got,direct),normalized_dirty_error=metric(dirty,refdirty),
            sameprocess_direct_apply_over_NUFFT_apply=out['GPU_direct_sameprocess']['apply_wall']['median_s']/float(np.median(vals)),
            observed_stage_device_bytes_above_preplan=int(max(baseline[0]-after_setup[0],baseline[0]-after_apply[0],0)),memory_method='cudaMemGetInfo before plan/after setpts/after apply, total device observation; excludes unsampled transient peaks; other GPU processes not subtracted')
        arrays['eps_'+str(eps)]=got;del plan;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    out.update(status='completed',results=results,GPU_occupied_s=time.time()-gstart,T05_GPU_direct_apply_median_s_reference_only=old['datasets'][name]['GPU']['apply_wall']['median_s'],input_V_sha256=hashlib.sha256(y.tobytes()).hexdigest(),T05_report_sha256=sha(T/'metrics/matched.json'),script_sha256=sha(Path(__file__)),comparison='same physical GPU, same process and frozen input; direct and each cuFINUFFT eps1warmup+10apply, setup and transfer timing separated')
    np.savez_compressed(O/'responses.npz',**arrays)
except Exception as e:out.update(status='failed',error=repr(e),traceback=traceback.format_exc())
out['elapsed_s']=time.time()-out['start'];(O/'gpu_roi.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(out,indent=2),flush=True)
