#include "common_mirror_tile_vissfast.hpp"
#include "orbit_online_mirror_tile_vissfast.cuh"
#include "viss_recon_kernel_mirror_tile_vissfast.cuh"


int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096);
  int day=to_int(get_arg(argc,argv,"--day","1"),1);
  int day_start=to_int(get_arg(argc,argv,"--day_start",std::to_string(day)), day);
  int day_count=to_int(get_arg(argc,argv,"--day_count","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);

  int blockage=to_int(get_arg(argc,argv,"--blockage","1"),1);
  int occ_mode=to_int(get_arg(argc,argv,"--occ_mode","2"),2);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","0"),0);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  std::string C_mode=get_arg(argc,argv,"--C_mode","bin");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 12345ULL);

  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  if(!gpus_s.empty()) setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

  int print_stats=to_int(get_arg(argc,argv,"--print_stats","1"),1);
  int viss_tile_pix=to_int(get_arg(argc,argv,"--viss_tile_pix","256"),256);
  if(viss_tile_pix!=256 && viss_tile_pix!=512){ std::cerr<<"ERROR --viss_tile_pix supports 256 or 512\n"; return 1; }

  if(btag!="1M" && btag!="10M" && btag!="30M"){
    std::cerr<<"ERROR --btag\n";
    return 1;
  }
  if(nside<=0){
    if(btag=="1M") nside=512;
    else if(btag=="10M") nside=4096;
    else nside=16384;
  }
  if(segs<=0){
    if(btag=="1M") segs=1;
    else if(btag=="10M") segs=10;
    else segs=30;
  }
  if(sky_dir.empty()) sky_dir=(btag=="1M")? "./earth_1Mhz" : (btag=="10M"? "./earth_10Mhz":"./earth_30Mhz");
  if(!out_dir.empty() && out_dir.back()!='/') out_dir.push_back('/');
  ensure_dir(out_dir);

  auto visible_gpus = parse_gpus(gpus_s);
  int requested_G = (int)visible_gpus.size();

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if(devCount <= 0){
    std::cerr<<"ERROR no visible CUDA devices after applying CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
    return 1;
  }

  int G = devCount;
  if(requested_G > 0 && requested_G != devCount){
    std::cerr<<"WARNING requested "<<requested_G
             <<" GPUs via --gpus="<<gpus_s
             <<", but CUDA sees "<<devCount
             <<". Use visible count = "<<devCount<<".\n";
  }

  std::vector<int> gpus(G);
  for(int i=0;i<G;i++) gpus[i] = i;

  std::cout<<"CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
  std::cout<<"Visible logical GPUs: ";
  for(int i=0;i<G;i++){std::cout<<gpus[i]; if(i+1<G) std::cout<<",";}
  std::cout<<"\n";

  long long npix=12LL*(long long)nside*(long long)nside;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix
           <<" day_start="<<day_start<<" day_count="<<day_count<<" segs="<<segs<<"\n";
  std::cout<<"blockage="<<blockage<<" occ_mode="<<occ_mode
           <<"  [Viss stage: blockage=ON, phase_correct=ON]\n";
  std::cout<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<" C_mode="<<C_mode<<"\n";

  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float bl_max=100e3f;

  std::cout<<"theta="<<theta<<" phi="<<phi<<" cosphi="<<cosphi<<" lamda="<<lamda<<"\n";

  HostTimer t_io;
  t_io.tic();
  std::vector<float> hB(npix);
  std::string used_B_path;
  if(!load_B_auto(sky_dir, btag, hB.data(), npix, B_mode, &used_B_path)){
    std::cerr<<"ERROR reading B in mode="<<B_mode<<" from "<<sky_dir<<"\n";
    return 1;
  }
  std::cout<<"Loaded B from "<<used_B_path<<" in "<<t_io.toc_s()<<" s\n";

  float lambda_m = lamda;
  int OrbitRes = (int)std::ceil((double)(2.0 * M_PI) * (double)(100e3 / (double)lambda_m));
  int ProcessionCount = round_away_from_zero_host(24.0f / ORBIT_HOURS);
  int segLen = OrbitRes / 3;
  int T = ProcessionCount * segLen;
  int N = T * SIGNED_BASELINES_PER_T;

  if(segs > T) segs = T;

  std::vector<int> seg_t0(segs), seg_tlen(segs);
  int baseT = T / segs;
  int remT = T % segs;
  int curT = 0;
  for(int k=0;k<segs;k++){
    int len = baseT + (k<remT?1:0);
    seg_t0[k]=curT;
    seg_tlen[k]=len;
    curT += len;
  }

  int max_tlen = 0;
  for(int k=0;k<segs;k++) max_tlen = std::max(max_tlen, seg_tlen[k]);
  int max_segN = max_tlen * SIGNED_BASELINES_PER_T;
  int max_segN_half = max_tlen * UNIQUE_BASELINES_PER_T;

  std::cout << "Online orbit plan: OrbitRes=" << OrbitRes
            << " ProcessionCount=" << ProcessionCount
            << " segLen=" << segLen
            << " T=" << T
            << " N=" << N << "\n";
  std::cout << "Segment plan: max_tlen=" << max_tlen
            << " max_segN=" << max_segN
            << " max_segN_half=" << max_segN_half << "\n";

  std::vector<GpuCtx> ctx(G);
  long long chunk_size=(npix + G - 1)/G;

  HostTimer t_init;
  t_init.tic();
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    int dev=gpus[gi];
    CHECK_CUDA(cudaSetDevice(dev));

    long long pix0=(long long)gi*chunk_size;
    long long pix1=std::min(npix,pix0+chunk_size);
    long long n_chunk=std::max(0LL,pix1-pix0);

    ctx[gi].dev=dev;
    ctx[gi].pix0=pix0;
    ctx[gi].pix1=pix1;
    ctx[gi].n_chunk=n_chunk;
    ctx[gi].N=N;
    ctx[gi].N_half=max_segN_half;
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].compute_stream,cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].xfer_stream,cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_nm1,(size_t)n_chunk*sizeof(float)));
    ctx[gi].ntile = (int)((n_chunk + viss_tile_pix - 1) / viss_tile_pix);
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx,(size_t)ctx[gi].ntile*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy,(size_t)ctx[gi].ntile*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz,(size_t)ctx[gi].ntile*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA,(size_t)ctx[gi].ntile*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA,(size_t)ctx[gi].ntile*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,hB.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].xfer_stream));

    int BLOCK=256;
    int grid=(int)((n_chunk+BLOCK-1)/BLOCK);
    pix2lmn_nest_kernel<<<grid,BLOCK,0,ctx[gi].xfer_stream>>>(
      nside, (unsigned int)pix0, (int)n_chunk, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    CHECK_CUDA(cudaPeekAtLastError());
    int grid_nm1=(int)((n_chunk+255)/256);
    build_nm1_kernel<<<grid_nm1,256,0,ctx[gi].xfer_stream>>>(ctx[gi].d_n, ctx[gi].d_nm1, ctx[gi].n_chunk);
    if(viss_tile_pix==256){
      build_tile_cone_meta_kernel<256><<<ctx[gi].ntile, 256, 0, ctx[gi].xfer_stream>>>(
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
        ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
        ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile);
    }else{
      build_tile_cone_meta_kernel<512><<<ctx[gi].ntile, 256, 0, ctx[gi].xfer_stream>>>(
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
        ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
        ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile);
    }
    CHECK_CUDA(cudaPeekAtLastError());

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      CHECK_CUDA(cudaMalloc(&sg.d_pos, (size_t)max_tlen * (3*SATNUM) * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_satx, (size_t)max_tlen * SATNUM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_saty, (size_t)max_tlen * SATNUM * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_satz, (size_t)max_tlen * SATNUM * sizeof(float)));

      CHECK_CUDA(cudaMalloc(&sg.d_u,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_v,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_w,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x1,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y1,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z1,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x2,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y2,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z2,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn1,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn2,(size_t)max_segN*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_Vpart,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&sg.d_Viss,(size_t)max_segN*sizeof(float2)));
      CHECK_CUDA(cudaMallocHost(&sg.h_Vpart,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaEventCreateWithFlags(&sg.ready, cudaEventDisableTiming));
    }

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cseg,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc,0,(size_t)n_chunk*sizeof(float),ctx[gi].xfer_stream));
    if(blockage==1){
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wacc,(size_t)n_chunk*sizeof(uint32_t)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wseg,(size_t)n_chunk*sizeof(uint32_t)));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wacc,0,(size_t)n_chunk*sizeof(uint32_t),ctx[gi].xfer_stream));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_keys_buf,(size_t)max_segN_half*sizeof(int)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_repidx_buf,(size_t)max_segN_half*sizeof(int)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_ugu_buf,(size_t)max_segN_half*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_vgu_buf,(size_t)max_segN_half*sizeof(float)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_repV_buf,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_repP1_buf,(size_t)max_segN_half*sizeof(float4)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_repP2_buf,(size_t)max_segN_half*sizeof(float4)));
    }
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)(1<<20)*sizeof(float)));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
  }
  std::cout<<"Init GPUs in "<<t_init.toc_s()<<" s\n";
  hB.clear(); hB.shrink_to_fit();

  if(gen_gpu_index < 0 || gen_gpu_index >= G){
    std::cerr << "ERROR gen_gpu_index out of range: " << gen_gpu_index
              << " / " << G << " (visible GPUs)\n";
    return 1;
  }

  BaselineBroadcastInfo binfo;
  query_peer_access(binfo, gpus[gen_gpu_index], ctx);
  std::cout << "Baseline broadcast: gen_dev=" << gpus[gen_gpu_index]
            << " need_host_stage=" << binfo.need_host_stage << "\n";

  auto reset_daily_accumulators = [&](void){
    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc,0,(size_t)ctx[gi].n_chunk*sizeof(float),ctx[gi].xfer_stream));
      if(blockage==1) CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wacc,0,(size_t)ctx[gi].n_chunk*sizeof(uint32_t),ctx[gi].xfer_stream));
      CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
    }
  };

  for(int day_idx=0; day_idx<day_count; ++day_idx){
    int dayid = day_start + day_idx;
    std::cout << "===== Day " << dayid << " / " << (day_start + day_count - 1) << " =====\n";
    reset_daily_accumulators();

    std::ofstream ofsViss;
    if(write_viss){
      std::string fv=out_dir+"Viss"+std::to_string(dayid)+"day"+btag+".txt";
      ofsViss.open(fv);
      if(!ofsViss.is_open()){
        std::cerr << "ERROR open " << fv << "\n";
        return 1;
      }
      static thread_local std::vector<char> vissbuf(8<<20);
      ofsViss.rdbuf()->pubsetbuf(vissbuf.data(), vissbuf.size());
      std::cout << "Will stream phase-corrected Viss to " << fv << "\n";
    }

    std::ofstream ofsUVW, ofsXYZa, ofsXYZb;
    if(write_baseline_txt){
      std::string fuvw = out_dir + "uvw" + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy1 = out_dir + "xyza" + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy2 = out_dir + "xyzb" + std::to_string(dayid) + "day" + btag + ".txt";
      ofsUVW.open(fuvw); ofsXYZa.open(fxy1); ofsXYZb.open(fxy2);
      if(!ofsUVW.is_open() || !ofsXYZa.is_open() || !ofsXYZb.is_open()){
        std::cerr << "ERROR open generated baseline txt outputs\n";
        return 1;
      }
      static thread_local std::vector<char> bufUVW(8<<20), bufXYZa(8<<20), bufXYZb(8<<20);
      ofsUVW.rdbuf()->pubsetbuf(bufUVW.data(), bufUVW.size());
      ofsXYZa.rdbuf()->pubsetbuf(bufXYZa.data(), bufXYZa.size());
      ofsXYZb.rdbuf()->pubsetbuf(bufXYZb.data(), bufXYZb.size());
      std::cout << "Will stream generated uvw/xyza/xyzb to txt under " << out_dir << "\n";
    }

    OrbitGenCtx gen;
    HostTimer t_gen_init;
    t_gen_init.tic();
    orbit_gen_init(gen, gpus[gen_gpu_index], lambda_m, dayid, max_tlen, orbit_seed);
    std::cout << "Orbit generator initialized on dev=" << gen.dev << " in " << t_gen_init.toc_s() << " s\n";

    bool need_all_host = true;
    orbit_gen_make_segment_async(gen, 0, seg_t0[0], seg_tlen[0], need_all_host);
    broadcast_segment_to_gpus_async(gen, 0, binfo, ctx, need_all_host);
    if(segs > 1){
      orbit_gen_make_segment_async(gen, 1, seg_t0[1], seg_tlen[1], need_all_host);
    }

    HostTimer t_rec;
    t_rec.tic();
    long long W_scalar_total=0;

    for(int s=0;s<segs;s++){
      int cur = s & 1;
      int nxt = cur ^ 1;
      orbit_gen_wait_slot(gen, cur);
      OrbitSegSlot &gcur = gen.slots[cur];
      int t0 = gcur.t0;
      int tlen = gcur.tlen;
      int segN = gcur.segN;
      int segN_half = tlen * UNIQUE_BASELINES_PER_T;
      if(segN<=0 || segN_half<=0) continue;

      if(!validate_uvw_halfsym_layout_ptr(gcur.h_u, gcur.h_v, gcur.h_w, segN)){
        std::cerr << "ERROR: generated segment uvw does not satisfy half-sym at seg " << s << " day " << dayid << "\n";
        return 1;
      }

      if(write_baseline_txt){
        write_txt_3cols_stream(ofsUVW,  gcur.h_u,  gcur.h_v,  gcur.h_w,  (size_t)segN);
        write_txt_3cols_stream(ofsXYZa, gcur.h_x1, gcur.h_y1, gcur.h_z1, (size_t)segN);
        write_txt_3cols_stream(ofsXYZb, gcur.h_x2, gcur.h_y2, gcur.h_z2, (size_t)segN);
      }

      HostTimer t_vseg;
      t_vseg.tic();
      const int BLOCKV=256;
      int gridV=(segN_half + BLOCKV - 1)/BLOCKV;
      size_t shmem=(size_t)viss_tile_pix*5*sizeof(float);

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaStreamWaitEvent(stream, slot.ready, 0));

        if(viss_tile_pix==256){
          viss_partial_all_halfsym_tilecone<256,true><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,ctx[gi].n_chunk,
            ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
            ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile,
            slot.d_u, slot.d_v, slot.d_w,
            slot.d_x1, slot.d_y1, slot.d_z1, slot.d_invn1,
            slot.d_x2, slot.d_y2, slot.d_z2, slot.d_invn2,
            segN_half, cosphi,
            slot.d_Vpart
          );
        }else{
          viss_partial_all_halfsym_tilecone<512,true><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,ctx[gi].n_chunk,
            ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
            ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile,
            slot.d_u, slot.d_v, slot.d_w,
            slot.d_x1, slot.d_y1, slot.d_z1, slot.d_invn1,
            slot.d_x2, slot.d_y2, slot.d_z2, slot.d_invn2,
            segN_half, cosphi,
            slot.d_Vpart
          );
        }
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaMemcpyAsync(slot.h_Vpart, slot.d_Vpart,
                                   (size_t)segN_half*sizeof(float2),
                                   cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));
      }

      std::vector<float2> hViss_half(segN_half);
      for(int i=0;i<segN_half;i++){
        float re=0, im=0;
        for(int gi=0; gi<G; ++gi){
          re += ctx[gi].slots[cur].h_Vpart[i].x;
          im += ctx[gi].slots[cur].h_Vpart[i].y;
        }
        hViss_half[i]=make_float2((float)re,(float)im);
      }
      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);

      std::vector<float2> hViss_seg;
      if(write_viss){
        hViss_seg.assign(segN, make_float2(0,0));
        expand_viss_halfsym_to_full(hViss_half, hViss_seg);
      }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaMemcpyAsync(slot.d_Viss, hViss_half.data(),
                                   (size_t)segN_half*sizeof(float2),
                                   cudaMemcpyHostToDevice, stream));
      }

      if(write_viss){
        for(int i=0;i<segN;i++) ofsViss << hViss_seg[i].x << " " << hViss_seg[i].y << "\n";
      }

      double t_vseg_s = t_vseg.toc_s();

      HostTimer t_rseg;
      t_rseg.tic();

      float fa=0, fb=0;
      double denom=0;
      compute_fa_fb_host_matlab_no_intercept(gcur.h_u, gcur.h_v, gcur.h_w, segN, fa, fb, denom);

      float umin,umax,vmin,vmax,wmin,wmax;
      if(print_stats) uvw_stats(gcur.h_u, gcur.h_v, gcur.h_w, segN, umin,umax,vmin,vmax,wmin,wmax);

      float lmax=std::sqrt(1.0f+fa*fa);
      float mmax=std::sqrt(1.0f+fb*fb);
      float lmmax=std::max(lmax,mmax);
      float temp_res=std::ceil(2.0f*bl_max/lamda*2.0f*lmmax);
      int RES=(int)(temp_res + 2 + 1 - std::fmod(temp_res,2.0f));
      int half=(RES-1)/2;
      float du=2.0f*bl_max/lamda/(RES-1);

      GroupArtifactsHost grp;
      CHECK_CUDA(cudaSetDevice(ctx[0].dev));
      build_groups_device0_half(ctx[0].dev, ctx[0].compute_stream,
                                ctx[0].slots[cur].d_u, ctx[0].slots[cur].d_v,
                                ctx[0].slots[cur].d_Viss,
                                segN_half, du, RES, half,
                                blockage,
                                occ_mode,
                                grp);
      if(grp.nuniq<=0){
        std::cout<<"[day "<<dayid<<" seg "<<(s+1)<<"/"<<segs<<"] nuniq<=0, skip\n";
        continue;
      }
      if(blockage==0) W_scalar_total += (long long)grp.nuniq;

      if(s+1 < segs){
        orbit_gen_wait_slot(gen, nxt);
        broadcast_segment_to_gpus_async(gen, nxt, binfo, ctx, need_all_host);
      }
      if(s+2 < segs){
        orbit_gen_make_segment_async(gen, cur, seg_t0[s+2], seg_tlen[s+2], need_all_host);
      }

      std::vector<int*> d_keys(G,nullptr);
      std::vector<float2*> d_vavg(G,nullptr);
      std::vector<int*> d_offsets(G,nullptr);
      std::vector<int*> d_idx(G,nullptr);
      std::vector<std::vector<float>> host_ugu(G), host_vgu(G);

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        d_keys[gi] = ctx[gi].d_keys_buf;
        CHECK_CUDA(cudaMemcpyAsync(d_keys[gi], grp.keys_unique.data(), (size_t)grp.nuniq*sizeof(int), cudaMemcpyHostToDevice, stream));

        if(blockage==0){
          d_vavg[gi] = ctx[gi].d_repV_buf;
          CHECK_CUDA(cudaMemcpyAsync(d_vavg[gi], grp.viss_avg.data(), (size_t)grp.nuniq*sizeof(float2), cudaMemcpyHostToDevice, stream));
        } else if(occ_mode==2 && !grp.rep_half_idx.empty()) {
          std::vector<int> h_ui(grp.nuniq), h_vi(grp.nuniq);
          host_ugu[gi].resize(grp.nuniq);
          host_vgu[gi].resize(grp.nuniq);
          for(int q=0; q<grp.nuniq; ++q){
            int key = grp.keys_unique[q];
            int tmp = key - 1;
            int U = tmp / RES;
            int V = tmp - U * RES;
            int ui = U - half;
            int vi = V - half;
            host_ugu[gi][q] = ui * du;
            host_vgu[gi][q] = vi * du;
          }
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_repidx_buf, grp.rep_half_idx.data(), (size_t)grp.nuniq*sizeof(int), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_ugu_buf, host_ugu[gi].data(), (size_t)grp.nuniq*sizeof(float), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_vgu_buf, host_vgu[gi].data(), (size_t)grp.nuniq*sizeof(float), cudaMemcpyHostToDevice, stream));
          int b0=256, g0=(grp.nuniq + b0 - 1)/b0;
          gather_rep_half_inputs<<<g0,b0,0,stream>>>(
            ctx[gi].d_repidx_buf, grp.nuniq,
            ctx[gi].slots[cur].d_Viss,
            ctx[gi].slots[cur].d_x1, ctx[gi].slots[cur].d_y1, ctx[gi].slots[cur].d_z1, ctx[gi].slots[cur].d_invn1,
            ctx[gi].slots[cur].d_x2, ctx[gi].slots[cur].d_y2, ctx[gi].slots[cur].d_z2, ctx[gi].slots[cur].d_invn2,
            ctx[gi].d_repV_buf, ctx[gi].d_repP1_buf, ctx[gi].d_repP2_buf
          );
          CHECK_CUDA(cudaPeekAtLastError());
        } else {
          CHECK_CUDA(cudaMalloc(&d_offsets[gi], (size_t)(grp.nuniq+1)*sizeof(int)));
          CHECK_CUDA(cudaMemcpyAsync(d_offsets[gi], grp.offsets.data(), (size_t)(grp.nuniq+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMalloc(&d_idx[gi], (size_t)segN_half*sizeof(int)));
          CHECK_CUDA(cudaMemcpyAsync(d_idx[gi], grp.idx_sorted.data(), (size_t)segN_half*sizeof(int), cudaMemcpyHostToDevice, stream));
        }
      }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];

        CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cseg, 0, (size_t)ctx[gi].n_chunk*sizeof(float), stream));
        if(blockage==1) CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wseg, 0, (size_t)ctx[gi].n_chunk*sizeof(uint32_t), stream));

        int BLOCK=256;
        int grid=(int)((ctx[gi].n_chunk + BLOCK - 1)/BLOCK);

        if(blockage==0){
          recon_seg_keys_avg_half<256><<<grid,BLOCK,0,stream>>>(
            ctx[gi].n_chunk,
            ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,
            d_keys[gi], d_vavg[gi],
            grp.nuniq,
            RES,half,du,fa,fb,
            ctx[gi].d_Cseg
          );
        } else if(occ_mode==2 && !grp.rep_half_idx.empty()) {
          recon_seg_blockage_half_rep<64><<<grid,BLOCK,0,stream>>>(
            ctx[gi].n_chunk,
            ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,
            d_keys[gi],
            ctx[gi].d_ugu_buf, ctx[gi].d_vgu_buf,
            ctx[gi].d_repV_buf,
            ctx[gi].d_repP1_buf, ctx[gi].d_repP2_buf,
            grp.nuniq,
            fa,fb,
            cosphi,
            ctx[gi].d_Cseg, ctx[gi].d_Wseg
          );
        } else {
          recon_seg_blockage_half<64><<<grid,BLOCK,0,stream>>>(
            ctx[gi].n_chunk,
            ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,
            d_keys[gi], d_offsets[gi], d_idx[gi],
            slot.d_Viss,
            slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,
            grp.nuniq,
            RES,half,du,fa,fb,
            cosphi, occ_mode,
            ctx[gi].d_Cseg, ctx[gi].d_Wseg
          );
        }
        CHECK_CUDA(cudaPeekAtLastError());

        int b=256;
        int g2=(int)((ctx[gi].n_chunk+b-1)/b);
        add_inplace<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].d_Cseg, ctx[gi].n_chunk);
        if(blockage==1) add_inplace_u32<<<g2,b,0,stream>>>(ctx[gi].d_Wacc, ctx[gi].d_Wseg, ctx[gi].n_chunk);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaStreamSynchronize(stream));
      }

      double t_rseg_s = t_rseg.toc_s();

      std::cout<<"[day "<<dayid<<" seg "<<(s+1)<<"/"<<segs<<"] t0="<<t0
               <<" segN="<<segN
               <<" segN_half="<<segN_half
               <<" VissTime="<<t_vseg_s<<"s"
               <<" ReconSegTime="<<t_rseg_s<<"s"
               <<" fa="<<fa<<" fb="<<fb
               <<" denom="<<denom
               <<" nuniq_halfgrid="<<grp.nuniq
               <<" RES="<<RES;
      if(print_stats){
        std::cout<<" u=["<<umin<<","<<umax<<"] v=["<<vmin<<","<<vmax<<"] w=["<<wmin<<","<<wmax<<"]";
      }
      std::cout<<"\n";
    }

    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].compute_stream;
      int b=256;
      int g2=(int)((ctx[gi].n_chunk+b-1)/b);
      if(blockage==0){
        float inv = (W_scalar_total>0)? (1.0f/(float)W_scalar_total) : 0.0f;
        scale_inplace<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].n_chunk, inv);
      } else {
        normalize_by_weight<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].d_Wacc, ctx[gi].n_chunk);
      }
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    std::cout<<"Recon done in "<<t_rec.toc_s()<<" s\n";

    HostTimer t_w;
    t_w.tic();
    std::string outC = out_dir+"C"+std::to_string(dayid)+"day"+btag + ((C_mode=="txt")? ".txt" : ".bin");
    if(C_mode=="txt"){
      std::ofstream ofs(outC);
      if(!ofs.is_open()){
        std::cerr<<"ERROR open "<<outC<<"\n";
        return 1;
      }
      static thread_local std::vector<char> outbuf(8<<20);
      ofs.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());
      const int CH=1<<20;
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for(long long off=0; off<ctx[gi].n_chunk; off+=CH){
          int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          for(int i=0;i<cur;i++) ofs<<ctx[gi].h_chunk[i]<<"\n";
        }
      }
      ofs.close();
    } else {
      std::ofstream ofs(outC, std::ios::binary);
      if(!ofs.is_open()){
        std::cerr<<"ERROR open "<<outC<<"\n";
        return 1;
      }
      const int CH=1<<20;
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for(long long off=0; off<ctx[gi].n_chunk; off+=CH){
          int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          ofs.write(reinterpret_cast<const char*>(ctx[gi].h_chunk), (std::streamsize)cur*sizeof(float));
        }
      }
      ofs.close();
    }
    std::cout<<"Wrote "<<outC<<" in "<<t_w.toc_s()<<" s\n";

    if(write_viss) ofsViss.close();
    if(write_baseline_txt){ ofsUVW.close(); ofsXYZa.close(); ofsXYZb.close(); }
    orbit_gen_destroy(gen);
  }

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);

    if(ctx[gi].d_repP2_buf) cudaFree(ctx[gi].d_repP2_buf);
    if(ctx[gi].d_repP1_buf) cudaFree(ctx[gi].d_repP1_buf);
    if(ctx[gi].d_repV_buf) cudaFree(ctx[gi].d_repV_buf);
    if(ctx[gi].d_vgu_buf) cudaFree(ctx[gi].d_vgu_buf);
    if(ctx[gi].d_ugu_buf) cudaFree(ctx[gi].d_ugu_buf);
    if(ctx[gi].d_repidx_buf) cudaFree(ctx[gi].d_repidx_buf);
    if(ctx[gi].d_keys_buf) cudaFree(ctx[gi].d_keys_buf);
    if(ctx[gi].d_Wseg) cudaFree(ctx[gi].d_Wseg);
    if(ctx[gi].d_Wacc) cudaFree(ctx[gi].d_Wacc);
    if(ctx[gi].d_Cseg) cudaFree(ctx[gi].d_Cseg);
    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      if(sg.ready) cudaEventDestroy(sg.ready);
      if(sg.h_Vpart) cudaFreeHost(sg.h_Vpart);
      if(sg.d_Viss) cudaFree(sg.d_Viss);
      if(sg.d_Vpart) cudaFree(sg.d_Vpart);
      if(sg.d_invn2) cudaFree(sg.d_invn2);
      if(sg.d_invn1) cudaFree(sg.d_invn1);
      if(sg.d_z2) cudaFree(sg.d_z2);
      if(sg.d_y2) cudaFree(sg.d_y2);
      if(sg.d_x2) cudaFree(sg.d_x2);
      if(sg.d_z1) cudaFree(sg.d_z1);
      if(sg.d_y1) cudaFree(sg.d_y1);
      if(sg.d_x1) cudaFree(sg.d_x1);
      if(sg.d_w) cudaFree(sg.d_w);
      if(sg.d_v) cudaFree(sg.d_v);
      if(sg.d_u) cudaFree(sg.d_u);
      if(sg.d_satz) cudaFree(sg.d_satz);
      if(sg.d_saty) cudaFree(sg.d_saty);
      if(sg.d_satx) cudaFree(sg.d_satx);
      if(sg.d_pos) cudaFree(sg.d_pos);
    }

    if(ctx[gi].d_tile_sinA) cudaFree(ctx[gi].d_tile_sinA);
    if(ctx[gi].d_tile_cosA) cudaFree(ctx[gi].d_tile_cosA);
    if(ctx[gi].d_tile_cz) cudaFree(ctx[gi].d_tile_cz);
    if(ctx[gi].d_tile_cy) cudaFree(ctx[gi].d_tile_cy);
    if(ctx[gi].d_tile_cx) cudaFree(ctx[gi].d_tile_cx);
    if(ctx[gi].d_nm1) cudaFree(ctx[gi].d_nm1);
    if(ctx[gi].d_n) cudaFree(ctx[gi].d_n);
    if(ctx[gi].d_m) cudaFree(ctx[gi].d_m);
    if(ctx[gi].d_l) cudaFree(ctx[gi].d_l);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].xfer_stream) cudaStreamDestroy(ctx[gi].xfer_stream);
    if(ctx[gi].compute_stream) cudaStreamDestroy(ctx[gi].compute_stream);
  }

  return 0;
}
