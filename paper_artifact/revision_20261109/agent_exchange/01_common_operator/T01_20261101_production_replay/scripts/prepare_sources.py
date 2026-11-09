#!/usr/bin/env python3
"""Copy SHA-frozen production source and add cache/sample hooks, leaving originals intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

EXPECTED = {
    'common_mirror_tile_vissfast.hpp':'0c6fa65037dae9cd48337fceac27c00bb00647af8e7b228539404438b35c5e0b',
    'main_recon_online_mirror_tile_vissfast.cu':'1daf59a7778c9258594d837b45a02d2013c03514a46a700d47ed4d34eda66990',
    'orbit_online_mirror_tile_vissfast.cuh':'417f6dee9c52583c27b39c28d0b33034625d88f47295a51d164c49c05fd1d345',
    'viss_recon_kernel_mirror_tile_vissfast.cuh':'d5794f2f13d26174342db7cdbc20cc4959685a7fede4fae9bd58b559d8cec320',
}

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('patch anchor count != 1: '+old[:100])
    return text.replace(old, new)

def main():
    p=argparse.ArgumentParser();p.add_argument('--fm',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    args=p.parse_args();source=args.root/'source';source.mkdir(parents=True,exist_ok=True)
    for name, expected in EXPECTED.items():
        data=(args.fm/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:raise RuntimeError('production SHA mismatch: '+name)
        (source/name).write_bytes(data)
    scripts=Path(__file__).resolve().parent
    for name in ('replay_common.hpp','replay.cu'):
        if (scripts/name).exists():shutil.copyfile(scripts/name,source/name)
    text=(source/'main_recon_online_mirror_tile_vissfast.cu').read_text()
    text=replace_once(text,'struct SegmentProfileRow {','#include "replay_common.hpp"\n\nstruct SegmentProfileRow {')
    text=replace_once(text,'  int write_output=to_int(',
        '  std::string cache_root=get_arg(argc,argv,"--cache_root","");\n'
        '  int cache_only=to_int(get_arg(argc,argv,"--cache_only","1"),1);\n'
        '  int replay_ns=to_int(get_arg(argc,argv,"--replay_ns","32768"),32768);\n'
        '  if(cache_root.empty() || replay_ns%256) {std::cerr<<"cache_root required; samples multiple256\\n";return 2;}\n'
        '  ensure_dir(cache_root);\n  int write_output=to_int(')
    text=replace_once(text,'      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);',
        '      std::vector<float2> replay_raw=hViss_half;\n'
        '      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);\n'
        '      CHECK_CUDA(cudaSetDevice(ctx[0].dev));\n'
        '      save_cache(cache_root,nside,dayid,s+1,G,gcur,ctx[0].slots[cur],replay_raw,hViss_half);')
    text=replace_once(text,'      std::vector<int*> d_keys(G,nullptr);',
        '      if(cache_only){std::cout<<"CACHE_ONLY_DONE day="<<dayid<<" seg="<<s+1<<"/"<<segs<<"\\n";continue;}\n'
        '      std::vector<int*> d_keys(G,nullptr);')
    anchor='    #pragma omp parallel for num_threads(G)\n    for(int gi=0; gi<G; ++gi){\n      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));\n      cudaStream_t stream=ctx[gi].compute_stream;\n      int b=256;'
    hook='''    if(!cache_only){
      auto ids=sample_ids(nside,replay_ns);
      std::vector<float> rawN(replay_ns,0.0f);std::vector<uint32_t> rawW(replay_ns,0);
      for(int gi=0;gi<G;gi++){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));auto di=device_copy(ids);float* dn=nullptr;uint32_t* dw=nullptr;
        CHECK_CUDA(cudaMalloc(&dn,replay_ns*sizeof(float)));CHECK_CUDA(cudaMalloc(&dw,replay_ns*sizeof(uint32_t)));
        CHECK_CUDA(cudaMemset(dn,0,replay_ns*sizeof(float)));CHECK_CUDA(cudaMemset(dw,0,replay_ns*sizeof(uint32_t)));
        gather_samples<<<(replay_ns+255)/256,256>>>(di,replay_ns,ctx[gi].pix0,ctx[gi].pix1,ctx[gi].d_Cacc,dn);
        gather_samples_u32<<<(replay_ns+255)/256,256>>>(di,replay_ns,ctx[gi].pix0,ctx[gi].pix1,ctx[gi].d_Wacc,dw);
        auto nn=host_copy(dn,replay_ns);auto ww=host_copy(dw,replay_ns);
        for(int j=0;j<replay_ns;j++)if(ids[j]>=(uint64_t)ctx[gi].pix0 && ids[j]<(uint64_t)ctx[gi].pix1){rawN[j]=nn[j];rawW[j]=ww[j];}
        CHECK_CUDA(cudaFree(di));CHECK_CUDA(cudaFree(dn));CHECK_CUDA(cudaFree(dw));
      }
      std::string prefix=cache_root+"/production_gate_day"+std::to_string(dayid);
      write_values(prefix+"_ids.u32",ids);write_values(prefix+"_N.f32",rawN);write_values(prefix+"_W.u32",rawW);
      std::cout<<"PRODUCTION_SAMPLED_GATE "<<prefix<<" samples="<<replay_ns<<"\\n";
    }
'''
    text=replace_once(text,anchor,hook+anchor)
    (source/'main_cache.cu').write_text(text)
    (args.root/'production_source_identity.json').write_text(json.dumps(EXPECTED,indent=2)+'\n')
    print('prepared independent source:',source)

if __name__=='__main__':main()
