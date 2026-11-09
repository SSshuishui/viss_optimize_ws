#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,subprocess,tempfile,sys
root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['python3',str(root/'scripts/prepare_source.py')],check=True,capture_output=True)
old=(root/'source/original_main.txt').read_text();new=(root/'source/main_production_regression.cu').read_text()
checks={}
upstream=root.parent/'T09_20261101_short30/scripts'
for p in (root/'source/production').iterdir():checks['original_header_'+p.name]=sha(p)==sha(upstream/p.name)
checks['validated_MODE4_header']=sha(root/'source/validated/candidate_generated.cuh')==sha(root.parent/'T09_20261101_consumed_rep_slab_gate/source/candidate_generated.cuh')
# Actual production inverse launch and all arithmetic arguments are byte-for-byte preserved.
for token in ['recon_seg_keys_avg_half<256>','recon_seg_blockage_half_rep<64>','recon_seg_blockage_half<64>','phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);','compute_fa_fb_host_matlab_no_intercept(gcur.h_u, gcur.h_v, gcur.h_w, segN, fa, fb, denom);','normalize_by_weight<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].d_Wacc, ctx[gi].n_chunk);']:
 a=old.index(token);b=old.index(';',a)+1;c=new.index(token);d=new.index(';',c)+1;checks['unchanged_'+token.split('(')[0]]=old[a:b]==new[c:d]
checks['all_586432_no_selected_forward']='segN_half==586432' in new and 'slab_candidate<256,true,4>' in new and 'slot.d_u,slot.d_v,slot.d_w' in new
checks['persistent_state_checkpoint']='cudaMemcpyAsync(d_state[gi][cur],read.payload.data()' in new and 'prodck::write(ck,h' in new
checks['once_final_partition_cast']=new.count('prod::cast_state<<<')==1 and 'for(long long slab_off=h.next' in new
checks['same_prefix_1G3_1G2']='h.next==(1ULL<<30)' in new and 'references.state(host_state' in new
checks['full_raw_before_normalize']=new.index('prod::scan(ctx,out_dir,"N")')<new.index('normalize_by_weight<<<')<new.index('prod::scan(ctx,out_dir,"C")')
checks['allmap_disabled']='write_output==0&&write_viss==0&&write_baseline_txt==0' in new
checks['fixed_uuid2_uuid3']='gpus_s==(stage=="prefix"?uuid2:uuid2+","+uuid3)' in new
with tempfile.TemporaryDirectory() as t:
 exe=Path(t)/'checkpoint_qa';flags=['-isystem','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1','-isysroot','/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk'] if sys.platform=='darwin' else [];subprocess.run(['clang++','-std=c++17','-O0',*flags,str(root/'scripts/test_checkpoint.cpp'),'-o',str(exe)],check=True,capture_output=True)
 z=subprocess.run([str(exe),str(Path(t)/'test')],check=True,capture_output=True,text=True);checks['checkpoint_CPU_8_fixtures']=True
out={'state':'passed' if all(checks.values()) else 'failed','checks':checks,'scope':'CPU source contracts/checkpoint IO only; CUDA compilation, allocation and numeric gates pending','source_sha256':{str(p.relative_to(root)):sha(p) for p in sorted((root/'source').rglob('*')) if p.is_file()},'checkpoint_stdout':z.stdout}
(root/'metrics/CPU_SOURCE_QA.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['checks'],indent=2));assert all(checks.values())
