#!/usr/bin/env python3
"""Local-only source preparation; deliberately never compiles or launches CUDA."""
import difflib,hashlib,json,re,shutil
from pathlib import Path

BEGIN='// DIAGNOSTIC_W_BEGIN\n'
END='// DIAGNOSTIC_W_END\n'
def mark(s):return BEGIN+s+'\n'+END
def kernel(text,name):
    start=text.index('__global__ void '+name+'(')
    brace=text.index('{',start);depth=0
    for end in range(brace,len(text)):
        if text[end]=='{':depth+=1
        elif text[end]=='}':
            depth-=1
            if depth==0:return start,brace,end+1,text[start:end+1]
    raise RuntimeError(name)
def stripped_body(s):
    s=re.sub(re.escape(BEGIN)+'.*?'+re.escape(END),'',s,flags=re.S)
    b=s.index('{');return re.sub(r'\s+','',s[b:])
def main():
    root=Path(__file__).resolve().parent.parent
    oldroot=root.parent/'T01_20261101_legacy3d_lineage/source/recon_fastpath_fixed_index_atomic'
    for name in ('common.hpp','orbit_online.cuh','viss_recon_kernel.cuh'):
        shutil.copyfile(oldroot/name,root/'source/original_combined'/name)
        if name!='viss_recon_kernel.cuh':shutil.copyfile(oldroot/name,root/'source/weighted_diagnostic'/name)
    original=(oldroot/'viss_recon_kernel.cuh').read_text();text=original;audit=[]
    for name in ('recon_3d_direct_tasklist_vv_real','recon_3d_direct_tasklist_mixed_real'):
        start,brace,end,part=kernel(text,name);oldpart=part
        part=part.replace('float* __restrict__ Cacc)', 'float* __restrict__ Cacc,\n    const float* __restrict__ diag_pairw,\n    double* __restrict__ diag_Wacc)',1)
        if name.endswith('_vv_real'):
            part=part.replace('__shared__ float2 sV[TILE_BL];','__shared__ float2 sV[TILE_BL];\n'+mark('static_assert(TILE_BL>0 && (TILE_BL&(TILE_BL-1))==0 && TILE_BL<=TILE_PIX,"diagnostic reduction requires power-of-two BL <= pixel block");\n__shared__ double diag_wsum[TILE_BL];'),1)
            part=part.replace('sV[t] = Viss_half[ih];','sV[t] = Viss_half[ih];\n'+mark('diag_wsum[t] = (double)diag_pairw[ih];'),1)
            part=part.replace('  __syncthreads();\n\n  float acc_re',mark('if(lane<TILE_BL && lane>=tileBL)diag_wsum[lane]=0;')+'  __syncthreads();\n'+mark('for(int stride=TILE_BL/2;stride>0;stride>>=1){if(lane<stride)diag_wsum[lane]+=diag_wsum[lane+stride];__syncthreads();}')+'\n  float acc_re',1)
            part=part.replace('if(active) atomicAdd(&Cacc[pix], acc_re);','if(active) atomicAdd(&Cacc[pix], acc_re);\n'+mark('if(active)atomicAdd(&diag_Wacc[pix],diag_wsum[0]);'),1)
        else:
            part=part.replace('__shared__ int nvv, nmix;','__shared__ int nvv, nmix;\n'+mark('static_assert(TILE_BL>0 && (TILE_BL&(TILE_BL-1))==0 && TILE_BL<=TILE_PIX,"diagnostic reduction requires power-of-two BL <= pixel block");\n__shared__ float diag_q[TILE_BL];\n__shared__ double diag_wsum[TILE_BL];'),1)
            part=part.replace('sV[t]=Viss_half[ih];','sV[t]=Viss_half[ih];\n'+mark('diag_q[t]=diag_pairw[ih];'),1)
            before='  float acc_re = 0.0f;'
            extra='if(lane<TILE_BL)diag_wsum[lane]=(lane<nvv)?(double)diag_q[(int)svv_idx[lane]]:0.0;\n__syncthreads();\nfor(int stride=TILE_BL/2;stride>0;stride>>=1){if(lane<stride)diag_wsum[lane]+=diag_wsum[lane+stride];__syncthreads();}'
            part=part.replace(before,mark(extra)+before,1)
            predicate='if(!operator_point_visible_pair_dirs(sdx1[t],sdy1[t],sdz1[t],sdx2[t],sdy2[t],sdz2[t],lp,mp,npv,cosphi)) continue;'
            assert part.count(predicate)==1
            independent_w='if(active){double diag_acc_w=diag_wsum[0];for(int ii=0;ii<nmix;ii++){int t=(int)smix_idx[ii];if(operator_point_visible_pair_dirs(sdx1[t],sdy1[t],sdz1[t],sdx2[t],sdy2[t],sdz2[t],lp,mp,npv,cosphi))diag_acc_w+=(double)diag_q[t];}atomicAdd(&diag_Wacc[pix],diag_acc_w);}'
            part=part.replace('if(active) atomicAdd(&Cacc[pix], acc_re);','if(active) atomicAdd(&Cacc[pix], acc_re);\n'+mark(independent_w),1)
        assert stripped_body(oldpart)==stripped_body(part),name+' N body changed after diagnostic markers stripped'
        audit.append({'kernel':name,'original_N_body_sha256':hashlib.sha256(stripped_body(oldpart).encode()).hexdigest(),'new_N_body_after_W_markers_removed_sha256':hashlib.sha256(stripped_body(part).encode()).hexdigest(),'N_body_unchanged_after_diagnostic_insertions_removed':True})
        text=text[:start]+part+text[end:]
    text=text.replace('  float* d_Cacc=nullptr;','  float* d_Cacc=nullptr;\n  double* d_diag_weighted_W=nullptr;\n  const float* d_diag_pairw=nullptr;',1)
    assert text.count('num_vv, gctx.d_Cacc);')==1 and text.count('num_mixed, gctx.d_Cacc);')==1
    text=text.replace('num_vv, gctx.d_Cacc);','num_vv, gctx.d_Cacc, gctx.d_diag_pairw, gctx.d_diag_weighted_W);').replace('num_mixed, gctx.d_Cacc);','num_mixed, gctx.d_Cacc, gctx.d_diag_pairw, gctx.d_diag_weighted_W);')
    (root/'source/weighted_diagnostic/viss_recon_kernel.cuh').write_text(text)
    (root/'source/add_diagnostic_W.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='original_combined/viss_recon_kernel.cuh',tofile='weighted_diagnostic/viss_recon_kernel.cuh')))
    (root/'metrics/N_source_identity.json').write_text(json.dumps({'state':'local_source_prepared_not_compiled_not_GPU_executed','checks':audit,'scope':'N kernel body textual identity after exclusively marked diagnostic W insertions are removed; binary scheduling/atomic addition order may differ and require numerical gates'},indent=2)+'\n')
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
