#!/usr/bin/env python3
"""Local-only source clone. Formal source/bin remain unchanged."""
from pathlib import Path
import hashlib,json

def main():
    root=Path(__file__).resolve().parent.parent
    formal=root.parent.parent/'07_performance_system/T07_20261101_legacy_inverse_comparison/scripts'
    original=(formal/'replay_3d_full.cu').read_text();s=original
    s=s.replace('../source/legacy_combined/','../source/weighted_diagnostic/')
    s=s.replace('int main(int argc,char**argv)', '#include "weighted_export_helpers.hpp"\n\nint main(int argc,char**argv)',1)
    s=s.replace('if(gpu.empty()', 'std::string grid_C_path=get_arg(argc,argv,"--sts_grid_C",""),actual_C_path=get_arg(argc,argv,"--sts_actual_C","");if(grid_C_path.empty()||actual_C_path.empty())throw std::runtime_error("prevalidated STS C files required for original raw-output metrics");\n if(gpu.empty()',1)
    s=s.replace('(size_t)np*16+', '(size_t)np*24+',1)
    s=s.replace('std::array<float*,5>tc;', 'auto diag_W=alloc<double>(np);CHECK_CUDA(cudaMemset(diag_W,0,np*sizeof(double)));\n std::array<float*,5>tc;',1)
    s=s.replace('ctx.d_Cacc=C;', 'ctx.d_Cacc=C;ctx.d_diag_weighted_W=diag_W;ctx.d_diag_pairw=q;',1)
    needle='double sample_s=elapsed(tsample);'
    extra='auto sampleWdev=alloc<double>(ids.size());diag_gather_W<<<32,256>>>(diag_W,did,8192,sampleWdev);std::vector<double>sampleW(8192);CHECK_CUDA(cudaMemcpy(sampleW.data(),sampleWdev,sampleW.size()*sizeof(double),cudaMemcpyDeviceToHost));write_vec(out+"/sample_W.f64",sampleW);'
    assert s.count(needle)==1;s=s.replace(needle,extra+needle,1)
    start=s.index(' std::ofstream fs(out+"/full_stats.json");')
    end=s.index('\n std::ofstream seg(',start)
    replacement=''' auto tdiag=Clock::now();std::ostringstream rawjson;rawjson<<std::setprecision(17)<<"{\\"dtype\\":\\"float32\\",\\"count\\":"<<np<<",\\"finite_count\\":"<<finite<<",\\"min\\":"<<vmin<<",\\"max\\":"<<vmax<<",\\"sha256\\":\\""<<hex_sha(digest)<<"\\"}";diag_finalize_export(C,diag_W,dn,np,out,rawjson.str(),grid_C_path,actual_C_path);scan_s+=elapsed(tdiag);'''
    s=s[:start]+replacement+s[end:]
    s=s.replace('legacy_combined_fastpath','full_corrected_dcf_weighted_reference')
    s=s.replace('no hidden D','rawN has no hidden D; exported C uses explicitly recorded float64 W')
    s=s.replace('raw weighted N; diagnostic W is external previously validated, no full-W computation in this loop','rawN plus same-predicate float64W; diagnostic C export after raw scans')
    s=s.replace('\\"full_map_written\\":false','\\"temporary_C_written\\":true')
    (root/'scripts/replay_corrected_dcf_full.cu').write_text(s)
    original_sts=(formal/'replay_sts_full.cu').read_text();t=original_sts
    t=t.replace('scan_array(const T* device_ptr,long long n)', 'scan_array(const T* device_ptr,long long n,const std::string& binary_export="")',1)
    t=t.replace('SHA256_CTX context;SHA256_Init(&context);', 'std::ofstream exported;if(!binary_export.empty()){exported.open(binary_export,std::ios::binary);if(!exported.good())throw std::runtime_error("C export open");}\n  SHA256_CTX context;SHA256_Init(&context);',1)
    t=t.replace('SHA256_Update(&context,part.data(),take*sizeof(T));', 'SHA256_Update(&context,part.data(),take*sizeof(T));if(exported.is_open()){exported.write((char*)part.data(),take*sizeof(T));if(!exported.good())throw std::runtime_error("C export write");}',1)
    t=t.replace('const std::string map_stats=scan_array(N,np);', 'const std::string map_stats=scan_array(N,np,out+"/sts_"+phase+"_C.f32");',1)
    t=t.replace('\\"output_map_written\\":false','\\"temporary_C_written\\":true')
    (root/'scripts/replay_sts_export.cu').write_text(t)
    audit={'state':'prepared_locally_not_compiled_not_GPU_executed','formal_3D_source_sha256':hashlib.sha256(original.encode()).hexdigest(),'formal_STS_source_sha256':hashlib.sha256(original_sts.encode()).hexdigest(),'clones':['scripts/replay_corrected_dcf_full.cu','scripts/replay_sts_export.cu'],'scope':'independent correctness/export clones; no edits to formal benchmark source or binary'}
    (root/'metrics/source_clone_identity.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)
if __name__=='__main__':main()
