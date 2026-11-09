#!/usr/bin/env python3
"""Deterministic, checked patch of the frozen original main; never edits production."""
from pathlib import Path
import difflib, hashlib
ROOT=Path(__file__).resolve().parents[1]
old=(ROOT/'source/original_main.txt').read_text();s=old
def replace(a,b):
 global s
 assert s.count(a)==1,(a[:100],s.count(a));s=s.replace(a,b)
replace('#include "common_mirror_tile_vissfast_streamonly.hpp"\n#include "orbit_online_mirror_tile_vissfast_streamonly.cuh"\n#include "viss_recon_kernel_mirror_tile_vissfast_streamonly.cuh"', '''#include "production/common_mirror_tile_vissfast_streamonly.hpp"
#include "allocation_tracking.hpp"
#include "production/orbit_online_mirror_tile_vissfast_streamonly.cuh"
#include "production/viss_recon_kernel_mirror_tile_vissfast_streamonly.cuh"
#include "validated/candidate_generated.cuh"
#include "regression_support.hpp"''')
replace('int main(int argc,char** argv){','''int main(int argc,char** argv){
  try {
  HostTimer process_timer;process_timer.tic();
  const std::string stage=get_arg(argc,argv,"--regression_stage","INVALID");
  const int reg_mode=to_int(get_arg(argc,argv,"--regression_mode","-1"),-1);
  const std::string reg_identity=get_arg(argc,argv,"--execution_identity","");
  const std::string refs_dir=get_arg(argc,argv,"--references","");
  prod::require((stage=="prefix"||stage=="capacity"||stage=="full")&&(reg_mode==0||reg_mode==4)&&reg_identity.size()==64&&!refs_dir.empty(),"regression CLI identity/stage");
  prod::pause_path=get_arg(argc,argv,"--pause_file","");
  std::signal(SIGTERM,prod::stop_handler);std::signal(SIGINT,prod::stop_handler);
  prod::References references(refs_dir);''')
replace('  auto visible_gpus = parse_gpus(gpus_s);\n  int requested_G = (int)visible_gpus.size();','''  const std::string uuid2="GPU-d26f44d0-c546-24e1-433c-49a053ecf0ee",uuid3="GPU-99fbd651-f080-aa22-391b-f33caa723578";
  prod::require(gpus_s==(stage=="prefix"?uuid2:uuid2+","+uuid3),"only frozen GPU2/3 UUIDs in fixed order allowed");
  prod::require(btag=="30M"&&nside==16384&&day_start==1&&day_count==1&&segs==10&&segment_limit==1&&sample_count==2048&&blockage==1&&occ_mode==2&&write_output==0&&write_viss==0&&write_baseline_txt==0&&B_mode=="bin"&&orbit_seed==42&&viss_tile_pix==256&&gen_gpu_index==0,"fixed full original production parameters");
  prod::require(sky_slab_pix==(1LL<<25)||sky_slab_pix==(1LL<<27),"frozen IO layouts only");
  prod::require(stage!="prefix"||sky_slab_pix==(1LL<<25),"prefix uses first original 2^25 pixels");
  int requested_G=stage=="prefix"?1:2;
  prodalloc::log.open(out_dir+"allocation.csv");prodalloc::log<<"logical,requested_bytes,rc,free_bytes,total_bytes\\n";
  prod::require(!prod::pause(),"PAUSE before CUDA work");''')
start=s.index('  int G = devCount;');end=s.index('\n  std::vector<int> gpus(G);',start)
s=s[:start]+'''  prod::require(devCount==requested_G,"visible CUDA count must equal frozen UUID list; no fallback");
  int G=devCount;
'''+s[end:]
replace('  std::vector<GpuCtx> ctx(G);','''  prod::require(npix==3221225472LL&&max_tlen==20944&&max_segN_half==586432&&OrbitRes==62832&&T==209440,"full 30MHz computational plan");
  std::vector<GpuCtx> ctx(G);
  std::vector<std::array<double2*,2>> d_state(G),d_comp(G);''')
replace('      CHECK_CUDA(cudaMalloc(&sg.d_Viss,(size_t)max_segN_half*sizeof(float2)));','''      CHECK_CUDA(cudaMalloc(&sg.d_Viss,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&d_state[gi][slot],(size_t)max_segN_half*sizeof(double2)));
      CHECK_CUDA(cudaMalloc(&d_comp[gi][slot],(size_t)max_segN_half*sizeof(double2)));''')
replace('    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));','''    if(stage!="prefix"){
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));''')
replace('    CHECK_CUDA(cudaMalloc(&ctx[gi].d_keys_buf, (size_t)max_segN_half*sizeof(int)));','''    }
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_keys_buf, (size_t)max_segN_half*sizeof(int)));''')
replace('    reset_daily_accumulators();','    if(stage!="prefix") reset_daily_accumulators();')
anchor='      HostTimer t_vseg; t_vseg.tic();'
replace(anchor,'''      uint64_t geom_hash=references.geometry(gcur,cosphi);
      prod::require(t0==0&&tlen==20944&&segN_half==586432,"first original segment/full half baseline identity");
      prod::memory(out_dir,G,stage!="prefix");
      HostTimer t_vseg; t_vseg.tic();''')
start=s.index('      #pragma omp parallel for num_threads(G)\n      for(int gi=0; gi<G; ++gi){',s.index(anchor));end=s.index('\n      std::vector<float2> hViss_half',start)
launch='''slab_candidate<256,true,MODE><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart,STATE,COMP);'''
original='''viss_partial_all_halfsym_tilecone<256,true><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart);'''
prefix='''
      if(stage=="prefix"){
        int gi=0;CHECK_CUDA(cudaSetDevice(0));auto stream=ctx[0].compute_stream;auto&slot=ctx[0].slots[cur];
        CHECK_CUDA(cudaStreamWaitEvent(stream,slot.ready,0));int slabN=1<<25;
        int ntile_cur=prepare_slab_viss(0,0,slabN,stream);
        HostTimer prefix_original;prefix_original.tic();
        ORIGINAL
        std::vector<float2>a(segN_half),b(segN_half);CHECK_CUDA(cudaMemcpyAsync(a.data(),slot.d_Vpart,a.size()*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        double prefix_original_s=prefix_original.toc_s();
        CLONE
        CHECK_CUDA(cudaMemcpyAsync(b.data(),slot.d_Vpart,b.size()*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        bool clone=!std::memcmp(a.data(),b.data(),a.size()*8);prod::text(std::filesystem::path(out_dir)/"clone_gate.json",std::string("{\\"pass\\":")+(clone?"true":"false")+",\\"bitwise\\":"+(clone?"true":"false")+",\\"all_half_baselines\\":586432,\\"input_pixels\\":33554432}\\n");prod::require(clone,"original header versus MODE0 clone");
        for(int z=0;z<2;z++){CHECK_CUDA(cudaMemsetAsync(d_state[0][z],0,segN_half*16,stream));CHECK_CUDA(cudaMemsetAsync(d_comp[0][z],0,segN_half*16,stream));}
        HostTimer prefix_micro;prefix_micro.tic();
        MICRO
        std::vector<double2>state(segN_half);CHECK_CUDA(cudaMemcpyAsync(state.data(),d_state[0][cur],state.size()*16,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        double prefix_micro_s=prefix_micro.toc_s();
        references.state(state,std::filesystem::path(out_dir)/"prefix_micro_reference",true);
        // Exercise reset of the alternate state buffer using the same first-segment geometry.
        CHECK_CUDA(cudaMemsetAsync(d_state[0][cur^1],0x5a,segN_half*16,stream));CHECK_CUDA(cudaMemsetAsync(d_state[0][cur^1],0,segN_half*16,stream));
        ALT
        std::vector<double2>second(segN_half);CHECK_CUDA(cudaMemcpyAsync(second.data(),d_state[0][cur^1],second.size()*16,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));prod::require(!std::memcmp(state.data(),second.data(),state.size()*16),"alternate-buffer reset gate");
        prod::text(std::filesystem::path(out_dir)/"prefix.json","{\\"pass\\":true,\\"all_half_baselines\\":586432,\\"input_pixels\\":33554432,\\"original_plan_segments\\":10,\\"reset_buffers_bitwise\\":true,\\"scope\\":\\"one first-segment prefix and alternate state-buffer reset; no full C/W allocated; not complete production\\"}\\n");
        std::ostringstream costs;costs<<std::setprecision(12)<<"{\\"original_first33M_forward_s\\":"<<prefix_original_s<<",\\"micro_first33M_forward_s\\":"<<prefix_micro_s<<",\\"includes_D2H_sync\\":true,\\"total_internal_process_s\\":"<<process_timer.toc_s()<<"}\\n";prod::text(std::filesystem::path(out_dir)/"pilot_cost.json",costs.str());
        std::cout<<"PREFIX_GATE_PASS elapsed_s="<<process_timer.toc_s()<<std::endl;return 0;
      }
'''.replace('ORIGINAL',original).replace('CLONE',launch.replace('MODE','0').replace('STATE','d_state[0][cur]').replace('COMP','d_comp[0][cur]')).replace('MICRO',launch.replace('MODE','4').replace('STATE','d_state[0][cur]').replace('COMP','d_comp[0][cur]')).replace('ALT',launch.replace('MODE','4').replace('STATE','d_state[0][cur^1]').replace('COMP','d_comp[0][cur^1]'))
forward='''
      const std::filesystem::path cache=std::filesystem::path(out_dir)/"forward_raw.f32x2";
      bool cached=std::filesystem::exists(std::filesystem::path(out_dir)/"forward_committed.bin");
      std::vector<std::string> forward_errors(G);
      if(stage=="capacity"){
        for(int gi=0;gi<G;gi++){CHECK_CUDA(cudaSetDevice(gi));CHECK_CUDA(cudaMemsetAsync(ctx[gi].slots[cur].d_Viss,0,segN_half*8,ctx[gi].compute_stream));CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));}
      }else if(!cached){
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
       try{
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));auto stream=ctx[gi].compute_stream;auto&slot=ctx[gi].slots[cur];CHECK_CUDA(cudaStreamWaitEvent(stream,slot.ready,0));
        // State starts at zero once per segment / physical partition. Dcomp is identically zero in MODE4.
        CHECK_CUDA(cudaMemsetAsync(d_state[gi][cur],0,segN_half*16,stream));CHECK_CUDA(cudaMemsetAsync(d_comp[gi][cur],0,segN_half*16,stream));
        prodck::Header h;std::memcpy(h.identity,reg_identity.c_str(),65);h.geometry_hash=geom_hash;h.npix=npix;h.nside=nside;h.day=dayid;h.seed=orbit_seed;h.t0=t0;h.tlen=tlen;h.mode=reg_mode;h.gi=gi;h.G=G;h.pix0=ctx[gi].pix0;h.n_chunk=ctx[gi].n_chunk;h.slab=sky_slab_pix;h.N_half=segN_half;
        auto ck=std::filesystem::path(out_dir)/("forward_gpu"+std::to_string(gi)+".ck");
        std::vector<double2> host_state(segN_half);
        if(std::filesystem::exists(ck)){auto read=prodck::read(ck,h);h=read.header;if(reg_mode==4)CHECK_CUDA(cudaMemcpyAsync(d_state[gi][cur],read.payload.data(),read.payload.size(),cudaMemcpyHostToDevice,stream));else std::memcpy(partial_gpu[gi].data(),read.payload.data(),read.payload.size());CHECK_CUDA(cudaStreamSynchronize(stream));if(reg_mode==4&&gi==0&&h.next>=(1ULL<<30)){if(h.next==(1ULL<<30)){std::memcpy(host_state.data(),read.payload.data(),read.payload.size());references.state(host_state,std::filesystem::path(out_dir)/"same_prefix_allrep_gate",false);}else prod::require(std::filesystem::exists(std::filesystem::path(out_dir)/"same_prefix_allrep_gate.json"),"checkpoint passed oracle boundary without accepted gate; audit required");}}
        for(long long slab_off=h.next;slab_off<ctx[gi].n_chunk;slab_off+=sky_slab_pix){
          if(prod::pause())throw std::runtime_error("paused at saved forward slab boundary");
          int slabN=(int)std::min<long long>(sky_slab_pix,ctx[gi].n_chunk-slab_off);
          int ntile_cur=prepare_slab_viss(gi,slab_off,slabN,stream);
          if(reg_mode==0){ORIGINAL}else{MICRO}
          CHECK_CUDA(cudaPeekAtLastError());
          if(reg_mode==0){
            CHECK_CUDA(cudaMemcpyAsync(slot.h_Vpart,slot.d_Vpart,segN_half*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
            // This is exactly the original host FP32 per-IO-slab += path.
            for(int i=0;i<segN_half;i++){partial_gpu[gi][i].x+=slot.h_Vpart[i].x;partial_gpu[gi][i].y+=slot.h_Vpart[i].y;}
          }else{CHECK_CUDA(cudaMemcpyAsync(host_state.data(),d_state[gi][cur],segN_half*16,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));}
          h.next=slab_off+slabN;h.nslabs++;
          prodck::write(ck,h,reg_mode==4?(void*)host_state.data():(void*)partial_gpu[gi].data(),prodck::payload_bytes(h));
          if(reg_mode==4&&gi==0&&h.next==(1ULL<<30))references.state(host_state,std::filesystem::path(out_dir)/"same_prefix_allrep_gate",false);
          std::cout<<"FORWARD_CHECKPOINT mode="<<reg_mode<<" gpu="<<gi<<" pixels="<<h.next<<'/'<<h.n_chunk<<std::endl;
        }
        if(reg_mode==4){
          // The persistent DP prefix must be cast once, at physical-partition completion.
          prod::cast_state<<<gridV,BLOCKV,0,stream>>>(d_state[gi][cur],slot.d_Vpart,segN_half);
          CHECK_CUDA(cudaMemcpyAsync(partial_gpu[gi].data(),slot.d_Vpart,segN_half*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        }
       }catch(const std::exception&e){forward_errors[gi]=e.what();}
      }
      for(const auto&e:forward_errors)if(!e.empty())throw std::runtime_error(e);
      }
'''.replace('ORIGINAL',original).replace('MICRO',launch.replace('MODE','4').replace('STATE','d_state[gi][cur]').replace('COMP','d_comp[gi][cur]'))
s=s[:start]+prefix+forward+s[end:]
replace('      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);','''      if(stage!="capacity"){
        if(cached){hViss_half=prod::load_forward(out_dir,prod::forward_identity(reg_identity,reg_mode,G,sky_slab_pix,geom_hash));prod::checkpoint_cleanup(out_dir,G);}
        else{prod::commit_forward(out_dir,prod::forward_identity(reg_identity,reg_mode,G,sky_slab_pix,geom_hash),hViss_half);prod::checkpoint_cleanup(out_dir,G);}
        for(const auto&v:hViss_half)prod::require(std::isfinite(v.x)&&std::isfinite(v.y),"full forward V finite gate");
      }
      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);
      if(stage!="capacity")prod::blob(std::filesystem::path(out_dir)/"forward_consumed.f32x2",hViss_half.data(),hViss_half.size()*8);''')
replace('      occupied_groups_total += (long long)grp.nuniq;','''      references.groups(grp,fa,fb,du,RES,half);
      prod::memory(out_dir,G,true);
      prod::text(std::filesystem::path(out_dir)/"group_gate.json","{\\"pass\\":true,\\"half_baselines\\":586432,\\"groups\\":86668,\\"full_keys_repidx_geometry_fit_bitwise\\":true}\\n");
      if(stage=="capacity"){prod::text(std::filesystem::path(out_dir)/"capacity.json","{\\"pass\\":true,\\"G\\":2,\\"npix\\":3221225472,\\"resident_C_W_bytes_each\\":12884901888,\\"full_geometry_slots\\":2,\\"original_full_CUB_group_executed\\":true}\\n");return 0;}
      occupied_groups_total += (long long)grp.nuniq;''')
# Inverse launches and all their original arithmetic remain byte-identical. Only a boundary stop is inserted.
replace('        for(long long slab_off=0; slab_off<ctx[gi].n_chunk; slab_off += sky_slab_pix){','''        for(long long slab_off=0; slab_off<ctx[gi].n_chunk; slab_off += sky_slab_pix){
          if(prod::pause()){CHECK_CUDA(cudaStreamSynchronize(stream));break;}''')
replace('      #pragma omp parallel for num_threads(G)\n      for(int gi=0; gi<G; ++gi){\n        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));\n        if(d_offsets[gi])', '''      if(prod::pause())throw std::runtime_error("paused inverse; explicit resume replays inverse from committed full forward V");
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        if(d_offsets[gi])''')
needle='''    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].compute_stream; int b=256; int g2=(int)((ctx[gi].n_chunk+b-1)/b);'''
replace(needle,'''    prod::scan(ctx,out_dir,"N");prod::samples(ctx,npix,chunk_size,out_dir,"N");
    prod::scan(ctx,out_dir,"W");prod::samples(ctx,npix,chunk_size,out_dir,"W");
'''+needle)
replace('    double segment_loop_total_s = t_rec.toc_s();','''    prod::scan(ctx,out_dir,"C");prod::samples(ctx,npix,chunk_size,out_dir,"C");
    prod::final_info(out_dir,reg_mode,G,sky_slab_pix,process_timer.toc_s());
    double segment_loop_total_s = t_rec.toc_s();''')
replace('      GpuSegSlot &sg = ctx[gi].slots[slot];\n      if(sg.ready)', '''      GpuSegSlot &sg = ctx[gi].slots[slot];
      if(d_state[gi][slot])cudaFree(d_state[gi][slot]);if(d_comp[gi][slot])cudaFree(d_comp[gi][slot]);
      if(sg.ready)''')
assert s.endswith('  return 0;\n}\n');s=s[:-len('  return 0;\n}\n')]+'''  return 0;
  }catch(const std::exception&e){std::cerr<<"REGRESSION_STOP "<<e.what()<<std::endl;return prod::pause()?75:1;}
}
'''
out=ROOT/'source/main_production_regression.cu';out.write_text(s)
(ROOT/'source/production_integration.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original_main.txt',tofile='main_production_regression.cu')))
print(out)
