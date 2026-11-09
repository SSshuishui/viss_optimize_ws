#!/usr/bin/env python3
"""Independent CPU phase/reduction and FP32 computational-mask checks on all reps."""
import argparse,ctypes,itertools,json
from pathlib import Path
import numpy as np
class Geometry(ctypes.LittleEndianStructure):
 _fields_=[('magic',ctypes.c_uint64),('nside',ctypes.c_uint32),('day',ctypes.c_uint32),('t0',ctypes.c_uint32),('tlen',ctypes.c_uint32),('nrec',ctypes.c_uint32),('fa',ctypes.c_float),('fb',ctypes.c_float),('du',ctypes.c_float),('cosphi',ctypes.c_float),('RES',ctypes.c_int32),('half',ctypes.c_int32)]
RECORD=np.dtype([(n,'<u4') for n in ('half_id','full_id','time','pair')]+[(n,'<f4') for n in ('u','v','w','x1','y1','z1','in1','x2','y2','z2','in2','cosphi')])
def fma32(a,b,c):return (np.asarray(a,dtype=np.float64)*np.asarray(b,dtype=np.float64)+np.asarray(c,dtype=np.float64)).astype(np.float32)
def fmseq(d,xyz,inv,order):
 a,b,c=order
 tmp=np.float32(d[c]*xyz[c]);tmp=fma32(d[b],xyz[b],tmp);tmp=fma32(d[a],xyz[a],tmp)
 return np.float32(tmp*inv)
def rel(a,b):return float(np.linalg.norm(np.asarray(a)-np.asarray(b))/max(np.linalg.norm(b),1e-300))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent);ap.add_argument('--case',default='m0_default');a=ap.parse_args();root=a.root.resolve();p=root/'references/rep_geometry.bin';buf=p.read_bytes();g=Geometry.from_buffer_copy(buf[:ctypes.sizeof(Geometry)]);r=np.frombuffer(buf[ctypes.sizeof(Geometry):],RECORD);assert len(r)==g.nrec==86668
 keys=np.fromfile(root/'references/full_keys.i32','<i4');repidx=np.fromfile(root/'references/rep_half_idx.i32','<i4');assert np.array_equal(repidx,r['half_id']);assert np.all(np.diff(keys)>0)
 tmp=keys.astype(np.int64)-1;U=tmp//g.RES;V=tmp-U*g.RES;ugu=np.float32((U-g.half).astype(np.float32)*np.float32(g.du));vgu=np.float32((V-g.half).astype(np.float32)*np.float32(g.du))
 dirs=(root/'references/sample_directions.bin').read_bytes();ids=np.frombuffer(dirs[:8192],'<u4');xyz=np.frombuffer(dirs[8192:],'<f4').reshape(3,2048)
 case=root/'runs'/a.case;outids=np.fromfile(case/'sample_ids.u32','<u4');rawN=np.fromfile(case/'sample_N.f32','<f4');rawW=np.fromfile(case/'sample_W.u32','<u4');assert np.array_equal(ids,outids)
 vis=np.fromfile(case/'forward_consumed.f32x2','<f4').reshape(586432,2)[repidx].astype(np.float64);probe=((2*np.arange(16)+1)*2048)//32
 permutations=[(2,0,1)]+[v for v in itertools.permutations(range(3)) if v!=(2,0,1)] # canonical mul(m*y),fma(l*x),fma(n*z)
 count_rows=[];numerators=[];phase32a=[];phase32b=[]
 endpoints=[(tuple(r[n] for n in ('x1','y1','z1')),r['in1']),(tuple(r[n] for n in ('x2','y2','z2')),r['in2'])]
 for ix in probe:
  d=xyz[:,ix];masks=[];counts=[]
  for order in permutations:
   mask=np.ones(86668,dtype=bool)
   for ep,inv in endpoints:mask &= fmseq(d,ep,inv,order)>=np.float32(g.cosphi)
   masks.append(mask);counts.append(int(mask.sum()))
  mask=masks[0];lp=np.float32(np.float64(g.fa)*np.float64(d[2])+np.float64(d[0]));mp=np.float32(np.float64(g.fb)*np.float64(d[2])+np.float64(d[1]));phi=2*np.pi*(ugu.astype(np.float64)*float(lp)+vgu.astype(np.float64)*float(mp));value=vis[:,0]*np.cos(phi)-vis[:,1]*np.sin(phi);numerators.append(float(np.sum(value[mask],dtype=np.longdouble)))
  twopi=np.float32(2*np.float32(np.pi));pa=np.float32(twopi*fma32(ugu,lp,np.float32(vgu*mp))).astype(float);pb=np.float32(twopi*fma32(vgu,mp,np.float32(ugu*lp))).astype(float)
  phase32a.append(float(np.sum((vis[:,0]*np.cos(pa)-vis[:,1]*np.sin(pa))[mask],dtype=np.longdouble)));phase32b.append(float(np.sum((vis[:,0]*np.cos(pb)-vis[:,1]*np.sin(pb))[mask],dtype=np.longdouble)))
  count_rows.append({'sample_index':int(ix),'global_id':int(ids[ix]),'GPU_W':int(rawW[ix]),'canonical_CPU_W':counts[0],'all_six_FP32_counts':counts})
 cp=np.asarray(numerators);gN=rawN[probe].astype(float);Nerr=rel(gN,cp);counts_ok=all(v['GPU_W']==v['canonical_CPU_W'] and v['GPU_W']>0 for v in count_rows)
 result={'state':'passed' if counts_ok and Nerr<=1e-3 else 'failed_oracle_gate','scope':'16 fixed nonempty expected global probes x all86668 representative groups; frozen GPU FP32 directions/endpoints and canonical FP32 computational mask, independent FP64 phase/long-double sum; not physical geometry validation','case':a.case,'probes':count_rows,'canonical_mask_order':'mul(m*y), fma(l*x), fma(n*z), mul(invn)','canonical_counts_exact_nonempty':counts_ok,'all_six_counts_agree':all(len(set(v['all_six_FP32_counts']))==1 for v in count_rows),'raw_N_relative_l2_vs_FP64_phase':Nerr,'raw_N_gate':1e-3,'raw_N_relative_l2_vs_phase32_order_a':rel(gN,phase32a),'raw_N_relative_l2_vs_phase32_order_b':rel(gN,phase32b),'GPU_N':gN.tolist(),'CPU_N_FP64_phase':cp.tolist(),'note':'If a gate fails retain it and diagnose arithmetic conventions; do not change thresholds post hoc.'}
 output=case/'CPU_INVERSE_ORACLE.json';output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('probes','GPU_N','CPU_N_FP64_phase')},indent=2))
if __name__=='__main__':main()
