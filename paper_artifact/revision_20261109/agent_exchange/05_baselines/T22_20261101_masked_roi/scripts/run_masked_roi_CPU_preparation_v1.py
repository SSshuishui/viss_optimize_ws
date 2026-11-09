#!/usr/bin/env python3
"""Exact-mask grouped cuFINUFFT vs same-process masked FP64 direct.

Default is CPU preparation only. CUDA requires explicit --allow-gpu --gpu and
--expected-uuid from the root agent's coordinated GPU window.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
import argparse
import ctypes
import gc
import hashlib
import json
import platform
import time
import traceback
import uuid
from pathlib import Path
import numpy as np
from mask_statistics import mask_rows


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)
def atomic_npz(path, **values):
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as stream: np.savez_compressed(stream, **values)
    tmp.replace(path)
def elapsed_stats(v):
    return dict(n=len(v), individual_s=list(map(float,v)), median_s=float(np.median(v)),
                sample_sd_s=float(np.std(v,ddof=1)), IQR_s=float(np.percentile(v,75)-np.percentile(v,25)))
def metric(c, r):
    d=c-r
    return dict(relative_l2=float(np.linalg.norm(d)/np.linalg.norm(r)), max_absolute=float(np.max(np.abs(d))),
                rmse=float(np.sqrt(np.mean(np.abs(d)**2))))
def dirty(v, D): return np.divide(2*v.real,D,out=np.zeros(len(D),dtype=np.float64),where=D>0)
def dirty_metrics(c,r,D):
    valid=D>0; cc=dirty(c,D)[valid]; rr=dirty(r,D)[valid]
    return dict(**metric(cc,rr),rmse_over_reference_std=float(np.sqrt(np.mean((cc-rr)**2))/np.std(rr)),
                valid=int(valid.sum()),invalid=int((~valid).sum()),candidate_zeros_retained=True)


def load_case(root, label, rows, cosphi):
    meta=json.loads((root/'cases'/f'{label}.json').read_text())
    with np.load(root/'cases'/f'face{meta["face"]:02d}_coordinates.npz',allow_pickle=False) as q:
        coords=q['lmn'][:meta['targets']].copy()
    t0=time.perf_counter()
    packed=mask_rows(coords,rows[:,3:6],rows[:,7:10],rows[:,[6,10]],cosphi)
    assert hashlib.sha256(packed.tobytes()).hexdigest()==meta['packed_mask_sha256']
    unique,inverse,counts=np.unique(packed,axis=0,return_inverse=True,return_counts=True)
    decoded=np.unpackbits(unique,axis=1,count=len(coords),bitorder='little')
    M=np.ascontiguousarray(np.unpackbits(packed,axis=1,count=len(coords),bitorder='little'))
    D=2*M.sum(axis=0,dtype=np.uint32)
    with np.load(root/'cases'/f'{label}.npz',allow_pickle=False) as q:
        assert np.array_equal(q['group_id_per_baseline'],inverse)
        assert np.array_equal(q['group_sizes'],counts) and np.array_equal(q['D'],D)
    assert np.array_equal(unique[inverse],packed)
    groups=[]
    for group in range(len(unique)):
        visible=np.flatnonzero(decoded[group]).astype(np.int32)
        members=np.flatnonzero(inverse==group).astype(np.int32)
        groups.append(dict(group=group,members=members,visible=visible))
    return meta,np.ascontiguousarray(coords,dtype=np.float64),M,D,groups,time.perf_counter()-t0


def cpu_oracle(root,label,identity,fn,s,t,c,M,D,groups):
    outdir=root/'cpu_reference';outdir.mkdir(exist_ok=True)
    jp=outdir/f'{label}.json'; npz=outdir/f'{label}.npz'
    if jp.exists():
        old=json.loads(jp.read_text())
        assert old['identity']==identity and old['status']=='completed', 'Existing CPU case identity/status mismatch'
        assert old['output_sha256']==sha(npz)
        with np.load(npz,allow_pickle=False) as q:
            assert np.array_equal(q['D'],D)
            return q['reference_complex'].copy(),old
    def apply(threads):
        out=np.empty(len(t),np.complex128)
        fn(len(s),len(t),s.ctypes.data,t.ctypes.data,c.ctypes.data,M.ctypes.data,threads,out.ctypes.data)
        return out
    t0=time.perf_counter(); ref=apply(16);cpu16=time.perf_counter()-t0
    t0=time.perf_counter(); serial=apply(1);cpu1=time.perf_counter()-t0
    assert np.array_equal(ref,serial), 'CPU thread-count output differs'
    # Independent algebraic factorization at 16 fixed directions; no NUFFT.
    probes=np.arange(16,dtype=np.int64)*(len(t)-1)//15
    grouped=np.zeros(len(probes),np.complex128)
    for g in groups:
        keep=np.isin(probes,g['visible'])
        if not keep.any():continue
        members=g['members']; phase=2*np.pi*(t[probes[keep]] @ s[members].T)
        grouped[keep]+=np.exp(1j*phase) @ c[members]
    algebra=metric(grouped,ref[probes]);assert algebra['relative_l2']<1e-10
    atomic_npz(npz,reference_complex=ref,D=D)
    result=dict(status='completed',identity=identity,CPU16_one_reference_apply_s=cpu16,CPU1_one_control_apply_s=cpu1,
                CPU1_CPU16_bitwise=True,grouped_algebra_16_probes=algebra,
                timing_scope='single correctness/setup observations, not repeated CPU benchmark',output_sha256=sha(npz))
    atomic_json(jp,result);return ref,result


def cuda_uuid():
    class CUuuid(ctypes.Structure): _fields_=[('bytes',ctypes.c_ubyte*16)]
    driver=ctypes.CDLL('libcuda.so.1'); dev=ctypes.c_int()
    driver.cuDeviceGet.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int]
    assert driver.cuDeviceGet(ctypes.byref(dev),0)==0
    uid=CUuuid(); lookup=getattr(driver,'cuDeviceGetUuid_v2',driver.cuDeviceGetUuid)
    lookup.argtypes=[ctypes.POINTER(CUuuid),ctypes.c_int]
    assert lookup(ctypes.byref(uid),dev.value)==0
    return 'GPU-'+str(uuid.UUID(bytes=bytes(uid.bytes)))


CUDA_SOURCE=r'''
extern "C" __global__ void direct_apply(int ns,int nt,const double* s,const double* t,
 const double2* c,const unsigned char* M,double2* out){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=nt)return;
 double ar=0,ai=0,x=t[3*p],y=t[3*p+1],z=t[3*p+2];
 for(int j=0;j<ns;j++){
  if(!M[(long long)j*nt+p])continue;
  double ph=6.283185307179586476925286766559*((s[3*j]*x+s[3*j+1]*y)+s[3*j+2]*z);
  double sn,cs;sincos(ph,&sn,&cs);double2 v=c[j];
  ar+=v.x*cs-v.y*sn;ai+=v.x*sn+v.y*cs;
 }out[p]=make_double2(ar,ai);
}
extern "C" __global__ void add_visible(int n,const int* index,const double2* values,double2* out){
 int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n)return;
 int k=index[p];out[k].x+=values[p].x;out[k].y+=values[p].y;
}
'''


def gpu_case(root,label,identity,meta,s,t,c,M,D,groups,ref,cp,cufinufft,kernel,scatter,repeat,limit):
    outdir=root/'gpu_results';outdir.mkdir(exist_ok=True)
    jp=outdir/f'{label}.json'; npz=outdir/f'{label}.npz'
    if jp.exists():
        old=json.loads(jp.read_text())
        assert old['identity']==identity and old['status'] in ('completed','completed_with_numerical_gate_failure'), 'Existing GPU case mismatch; do not overwrite'
        assert old['output_sha256']==sha(npz)
        return old
    started=time.time();result=dict(status='running',identity=identity,case=label,mask_statistics=meta)
    atomic_json(jp,result)
    def memory_guard():
        free,total=cp.cuda.runtime.memGetInfo();occupied=total-free
        if occupied>limit:raise MemoryError(f'Observed total device occupancy {occupied} exceeds declared {limit} bytes')
        return dict(free_bytes=int(free),total_bytes=int(total),occupied_bytes=int(occupied))
    def timed_apply(apply):
        apply();cp.cuda.Device().synchronize();wall=[];events=[]
        for _ in range(repeat):
            e1,e2=cp.cuda.Event(),cp.cuda.Event();t0=time.perf_counter();e1.record();apply();e2.record();e2.synchronize()
            wall.append(time.perf_counter()-t0);events.append(cp.cuda.get_elapsed_time(e1,e2)*.001)
        return dict(apply_wall=elapsed_stats(wall),apply_events=elapsed_stats(events))
    arrays={}
    try:
        memory_before=memory_guard();t0=time.perf_counter()
        ds,dt,gc_,gm=cp.asarray(s),cp.asarray(t),cp.asarray(c),cp.asarray(M)
        go=cp.empty(len(t),cp.complex128);cp.cuda.Device().synchronize()
        result['direct_input_H2D_s']=time.perf_counter()-t0
        result['memory_before']=memory_before
        args=(np.int32(len(s)),np.int32(len(t)),ds,dt,gc_,gm,go)
        def direct():kernel(((len(t)+127)//128,),(128,),args)
        direct_timing=timed_apply(direct)
        t0=time.perf_counter();got_direct=cp.asnumpy(go);result['direct_D2H_s']=time.perf_counter()-t0
        direct_err=metric(got_direct,ref)
        assert direct_err['relative_l2']<1e-10,'Masked GPU direct failed CPU gate'
        t0=time.perf_counter();direct_dirty=dirty(got_direct,D);normalization=time.perf_counter()-t0
        result['GPU_direct']=dict(**direct_timing,raw_error_vs_CPU=direct_err,dirty_error_vs_CPU=dirty_metrics(got_direct,ref,D),
                                 CPU_post_D2H_normalization_s=normalization)
        arrays['GPU_direct']=got_direct
        del ds,dt,gc_,gm,go,args;gc.collect();cp.get_default_memory_pool().free_all_blocks()
        results={}
        for eps in (1e-6,1e-9):
            plans=[];setup=0;group_h2d=0;memory_marks=[];t0=time.perf_counter()
            total_output=cp.empty(len(t),cp.complex128)
            cp.cuda.Device().synchronize();accumulator_setup=time.perf_counter()-t0
            for g in groups:
                if not len(g['visible']):continue
                members,visible=g['members'],g['visible']
                t0=time.perf_counter()
                sources=[cp.asarray(np.ascontiguousarray(2*np.pi*s[members,k])) for k in range(3)]
                targets=[cp.asarray(np.ascontiguousarray(t[visible,k])) for k in range(3)]
                coefficients=cp.asarray(np.ascontiguousarray(c[members]));indices=cp.asarray(visible)
                group_output=cp.empty(len(visible),cp.complex128);cp.cuda.Device().synchronize()
                group_h2d+=time.perf_counter()-t0;memory_marks.append(memory_guard())
                t0=time.perf_counter()
                plan=cufinufft.Plan(3,3,n_trans=1,eps=eps,isign=1,dtype='complex128')
                plan.setpts(*sources,*targets);cp.cuda.Device().synchronize();setup+=time.perf_counter()-t0
                memory_marks.append(memory_guard())
                plans.append(dict(plan=plan,coefficients=coefficients,indices=indices,out=group_output,
                                  all_visible=len(visible)==len(t),sources=sources,targets=targets))
            def apply_all(diagnostic=False):
                total_output.fill(0)
                stamps=[]
                for g in plans:
                    if diagnostic:
                        a,b,c0,d=[cp.cuda.Event() for _ in range(4)];a.record()
                    g['plan'].execute(g['coefficients'],out=g['out'])
                    if diagnostic:b.record();c0.record()
                    if g['all_visible']:cp.add(total_output,g['out'],out=total_output)
                    else:scatter(((len(g['indices'])+127)//128,),(128,),(np.int32(len(g['indices'])),g['indices'],g['out'],total_output))
                    if diagnostic:d.record();stamps.append((a,b,c0,d))
                return stamps
            timing=timed_apply(apply_all)
            t0=time.perf_counter();got=cp.asnumpy(total_output);d2h=time.perf_counter()-t0
            diag=apply_all(True);cp.cuda.Device().synchronize()
            execute=sum(cp.cuda.get_elapsed_time(a,b)*.001 for a,b,_,_ in diag)
            merge=sum(cp.cuda.get_elapsed_time(c0,d)*.001 for _,_,c0,d in diag)
            err=metric(got,ref);threshold=1e-4 if eps==1e-6 else 1e-7
            t0=time.perf_counter();normalized=dirty(got,D);normalization=time.perf_counter()-t0
            results[str(eps)]=dict(all_group_plan_setpts_s=setup,all_group_input_build_H2D_s=group_h2d,
                accumulator_setup_s=accumulator_setup,**timing,D2H_s=d2h,raw_error_vs_CPU=err,
                raw_error_vs_sameprocess_GPU_direct=metric(got,got_direct),dirty_error_vs_CPU=dirty_metrics(got,ref,D),
                CPU_post_D2H_normalization_s=normalization,
                numerical_gate_pass=err['relative_l2']<=threshold,numerical_gate_threshold=threshold,
                transform_group_count=len(plans),D_shared_exact=True,
                mask_application='Each exact mask group transforms only its visible target subset; GPU scatter/add is included in complete apply',
                separate_one_instrumented_diagnostic=dict(total_group_execute_event_s=execute,total_group_merge_event_s=merge,
                    scope='Additional instrumented iteration, not part of the ten formal applies; event instrumentation can change timings'),
                observed_memory_marks=memory_marks,memory_after_apply=memory_guard(),
                sameprocess_direct_apply_over_complete_masked_NUFFT_apply=result['GPU_direct']['apply_wall']['median_s']/timing['apply_wall']['median_s'])
            arrays['NUFFT_eps_'+str(eps)]=got
            for g in plans:del g['plan']
            plans.clear();del plan,sources,targets,coefficients,indices,group_output,total_output,g
            gc.collect();cp.get_default_memory_pool().free_all_blocks()
        atomic_npz(npz,**arrays,D=D)
        result.update(status='completed' if all(v['numerical_gate_pass'] for v in results.values()) else 'completed_with_numerical_gate_failure',
                      results=results,output_sha256=sha(npz),elapsed_s=time.time()-started)
    except Exception as exc:
        result.update(status='failed_no_automatic_retry',error=repr(exc),traceback=traceback.format_exc(),elapsed_s=time.time()-started)
    atomic_json(jp,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--case',action='append',help='Default all36; explicit subset must have its selection disclosed')
    p.add_argument('--allow-gpu',action='store_true')
    p.add_argument('--gpu',choices=['1','2','3'])
    p.add_argument('--expected-uuid')
    args=p.parse_args();root=args.root.resolve()
    if args.allow_gpu and (not args.gpu or not args.expected_uuid):p.error('CUDA requires --gpu and --expected-uuid from coordinated preflight')
    if args.allow_gpu:os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    protocol=json.loads((root/'PROTOCOL.json').read_text())
    z=np.load(root/'inputs_existing.npz',allow_pickle=False);rows=z['rows'];cosphi=np.float32(z['cosphi'])
    s=np.ascontiguousarray(rows[:,:3],dtype=np.float64)
    c=np.ascontiguousarray(rows[:,13].astype(np.float64)+1j*rows[:,14].astype(np.float64))
    libpath=Path(__file__).with_name('direct.so');lib=ctypes.CDLL(str(libpath));fn=lib.apply_direct
    I=ctypes.c_int;V=ctypes.c_void_p;fn.argtypes=[I,I,V,V,V,V,I,V];fn.restype=None
    census=json.loads((root/'mask_statistics.json').read_text());labels=[q['case'] for q in census['cases']]
    if args.case:
        assert all(label in labels for label in args.case)
        labels=[label for label in labels if label in args.case]
    identity=dict(protocol_sha256=sha(root/'PROTOCOL.json'),inputs_sha256=sha(root/'inputs_existing.npz'),
                  CPU_source_sha256=sha(Path(__file__).with_name('direct.cpp')),CPU_binary_sha256=sha(libpath),
                  driver_sha256=sha(Path(__file__).resolve()),phase='positive consumed-n 2pi*(u*l+v*m+w*n)',
                  W='unit half-baseline weights',D='2*sum exact frozen mask',mode='GPU' if args.allow_gpu else 'CPU_preparation')
    cp=cufinufft=kernel=scatter=None;global_GPU={}
    if args.allow_gpu:
        import cupy as cp
        import cufinufft
        actual_uuid=cuda_uuid();assert actual_uuid==args.expected_uuid,'GPU identity differs from guarded physical card'
        t0=time.perf_counter();kernel=cp.RawKernel(CUDA_SOURCE,'direct_apply',options=('--fmad=false',));kernel.compile()
        scatter=cp.RawKernel(CUDA_SOURCE,'add_visible',options=('--fmad=false',));scatter.compile();cp.cuda.Device().synchronize()
        global_GPU=dict(actual_uuid=actual_uuid,physical_selection=args.gpu,PCI_bus_id=cp.cuda.runtime.deviceGetPCIBusId(0),
                        JIT_s=time.perf_counter()-t0,cufinufft_version=cufinufft.__version__,shared_node=True,clocks_locked=False)
        sx=np.asarray([.2,.4]);zero=np.zeros(2);coef=np.asarray([1+2j,.3-.7j]);tt=np.asarray([.5,1.2])
        target=np.exp(1j*tt[:,None]*sx)@coef
        got=cp.asnumpy(cufinufft.nufft3d3(cp.asarray(sx),cp.asarray(zero),cp.asarray(zero),cp.asarray(coef),cp.asarray(tt),cp.asarray(zero),cp.asarray(zero),eps=1e-12,isign=1))
        global_GPU['positive_isign_gate']=metric(got,target);assert metric(got,target)['relative_l2']<1e-10
    summaries=[]
    for label in labels:
        assert sum(q.stat().st_size for q in root.rglob('*') if q.is_file())<104857600,'Retention exceeds100MiB'
        meta,t,M,D,groups,grouping_s=load_case(root,label,rows,cosphi)
        cpu_id={key:value for key,value in identity.items() if key!='mode'}
        ref,cpu=cpu_oracle(root,label,cpu_id,fn,s,t,c,M,D,groups)
        if args.allow_gpu:
            static_GPU={key:global_GPU[key] for key in ('actual_uuid','physical_selection','PCI_bus_id','cufinufft_version')}
            result=gpu_case(root,label,{**identity,**static_GPU},meta,s,t,c,M,D,groups,ref,cp,cufinufft,kernel,scatter,10,16*1024**3)
            result['CPU_mask_build_grouping_s']=grouping_s
            atomic_json(root/'gpu_results'/f'{label}.json',result)
            summaries.append(dict(case=label,status=result['status']))
            cp.cuda.Device().synchronize();gc.collect();cp.get_default_memory_pool().free_all_blocks()
        else:summaries.append(dict(case=label,status=cpu['status'],CPU_mask_build_grouping_s=grouping_s))
        print(json.dumps(summaries[-1]),flush=True)
    atomic_json(root/('GPU_execution.json' if args.allow_gpu else 'CPU_preparation.json'),
                dict(status='completed',host=platform.node(),identity=identity,GPU=global_GPU,cases=summaries,
                     requested_cases=labels,all36_census_preserved=True,new_forward=False))


if __name__=='__main__':main()
