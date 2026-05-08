#include "common.hpp"
#include "orbit_online.cuh"
#include "viss_recon_kernel.cuh"

int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096); // default nside for 10M
  int day=to_int(get_arg(argc,argv,"--day","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);

  int blockage=to_int(get_arg(argc,argv,"--blockage","1"),1);
  int occ_mode=to_int(get_arg(argc,argv,"--occ_mode","2"),2);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","0"),1);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 12345ULL);

  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  if(!gpus_s.empty()){
    setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);
  }

  int print_stats=to_int(get_arg(argc,argv,"--print_stats","1"),1);

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

  // 程序内部统一使用逻辑设备号 0..G-1
  std::vector<int> gpus(G);
  for(int i=0;i<G;i++) gpus[i] = i;

  std::cout<<"CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
  std::cout<<"Visible logical GPUs: ";
  for(int i=0;i<G;i++){std::cout<<gpus[i]; if(i+1<G) std::cout<<",";}
  std::cout<<"\n";

  long long npix=12LL*(long long)nside*(long long)nside;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix<<" day="<<day<<" segs="<<segs<<"\n";
  std::cout<<"blockage="<<blockage<<" occ_mode="<<occ_mode
           <<"  [Viss stage: blockage=ON, phase_correct=ON]\n";
  std::cout<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<"\n";

  // physical params
  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float bl_max=100e3f;

  std::cout<<"theta="<<theta<<" phi="<<phi<<" cosphi="<<cosphi<<" lamda="<<lamda<<"\n";

  // Load B only (txt/bin auto path)
  HostTimer t_io;
  t_io.tic();
  std::vector<float> hB(npix);
  std::string used_B_path;
  if(!load_B_auto(sky_dir, btag, hB.data(), npix, B_mode, &used_B_path)){
    std::cerr<<"ERROR reading B in mode="<<B_mode<<" from "<<sky_dir<<"\n";
    return 1;
  }
  std::cout<<"Loaded B from "<<used_B_path<<" in "<<t_io.toc_s()<<" s\n";

  // C=single(B.*s);  
  const double pix_area = 4.0 * M_PI / (double)npix;
  #pragma omp parallel for
  for (long long i = 0; i < npix; ++i) {
    hB[i] = (float)((double)hB[i] * pix_area);
  }
  std::cout << " pix_area=" << pix_area
            << " inv_pix_area=" << (1.0 / pix_area)
            << "\n";

  // Online orbit / baseline generation plan
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

  // init per GPU
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
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].stream,cudaStreamNonBlocking));

    // sky chunk: B from host, lmn generated directly on device
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)n_chunk*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,hB.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].stream));

    int BLOCK=256;
    int grid=(int)((n_chunk+BLOCK-1)/BLOCK);
    pix2lmn_nest_kernel<<<grid,BLOCK,0,ctx[gi].stream>>>(
      nside, (unsigned int)pix0, (int)n_chunk, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    CHECK_CUDA(cudaPeekAtLastError());

    // segment-local baseline arrays
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_u,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_v,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_w,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_x1,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_y1,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_z1,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_x2,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_y2,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_z2,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn1,(size_t)max_segN*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn2,(size_t)max_segN*sizeof(float)));

    // segment Viss buffers
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Vpart,(size_t)max_segN_half*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Viss,(size_t)max_segN*sizeof(float2)));
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_Vpart,(size_t)max_segN_half*sizeof(float2)));

    // recon buffers
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cseg,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc,0,(size_t)n_chunk*sizeof(float),ctx[gi].stream));
    if(blockage==1){
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wacc,(size_t)n_chunk*sizeof(uint32_t)));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wseg,(size_t)n_chunk*sizeof(uint32_t)));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wacc,0,(size_t)n_chunk*sizeof(uint32_t),ctx[gi].stream));
    }
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)(1<<20)*sizeof(float)));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));
  }
  std::cout<<"Init GPUs in "<<t_init.toc_s()<<" s\n";

  // release host B to save RAM
  hB.clear(); hB.shrink_to_fit();

  if(gen_gpu_index < 0 || gen_gpu_index >= G){
    std::cerr << "ERROR gen_gpu_index out of range: " << gen_gpu_index
              << " / " << G << " (visible GPUs)\n";
    return 1;
  }
  OrbitGenCtx gen;
  orbit_gen_init(gen, gpus[gen_gpu_index], lambda_m, day, max_tlen, orbit_seed);
  std::cout << "Orbit generator initialized on dev=" << gen.dev << "\n";

  BaselineBroadcastInfo binfo;
  query_peer_access(binfo, gen.dev, ctx);
  std::cout << "Baseline broadcast: gen_dev=" << binfo.gen_dev
            << " need_host_stage=" << binfo.need_host_stage << "\n";

  // ---------------- Viss output file (streaming by segment) ----------------
  std::ofstream ofsViss;
  if(write_viss){
    std::string fv=out_dir+"Viss"+std::to_string(day)+"day"+btag+".txt";
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
    std::string fuvw = out_dir + "uvw" + std::to_string(day) + "day" + btag + ".txt";
    std::string fxy1 = out_dir + "xyza" + std::to_string(day) + "day" + btag + ".txt";
    std::string fxy2 = out_dir + "xyzb" + std::to_string(day) + "day" + btag + ".txt";
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

  // ---------------- Segmented reconstruction ----------------
  HostTimer t_rec;
  t_rec.tic();
  long long W_scalar_total=0;

  for(int s=0;s<segs;s++){
    int t0=seg_t0[s];
    int tlen=seg_tlen[s];
    int segN=tlen*SIGNED_BASELINES_PER_T;
    int segN_half=tlen*UNIQUE_BASELINES_PER_T;
    if(segN<=0 || segN_half<=0) continue;

    bool need_all_host = write_baseline_txt || binfo.need_host_stage;
    orbit_gen_make_segment(gen, t0, tlen, need_all_host);

    if(!validate_uvw_halfsym_layout_ptr(gen.h_u, gen.h_v, gen.h_w, segN)){
      std::cerr << "ERROR: generated segment uvw does not satisfy half-sym at seg " << s << "\n";
      return 1;
    }

    if(write_baseline_txt){
      write_txt_3cols_stream(ofsUVW,  gen.h_u,  gen.h_v,  gen.h_w,  (size_t)segN);
      write_txt_3cols_stream(ofsXYZa, gen.h_x1, gen.h_y1, gen.h_z1, (size_t)segN);
      write_txt_3cols_stream(ofsXYZb, gen.h_x2, gen.h_y2, gen.h_z2, (size_t)segN);
    }

    broadcast_segment_to_gpus(gen, binfo, ctx, segN, need_all_host);

    // ---------- segment Viss compute ----------
    HostTimer t_vseg;
    t_vseg.tic();
    constexpr int TILE_PIX=256;
    const int BLOCKV=256;
    int gridV=(segN_half + BLOCKV - 1)/BLOCKV;
    size_t shmem=(size_t)TILE_PIX*4*sizeof(float);

    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].stream;

      viss_partial_all_halfsym<TILE_PIX,true><<<gridV,BLOCKV,shmem,stream>>>(
        ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].n_chunk,
        ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
        ctx[gi].d_x1, ctx[gi].d_y1, ctx[gi].d_z1, ctx[gi].d_invn1,
        ctx[gi].d_x2, ctx[gi].d_y2, ctx[gi].d_z2, ctx[gi].d_invn2,
        segN_half, cosphi,
        ctx[gi].d_Vpart
      );
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaMemcpyAsync(
        ctx[gi].h_Vpart,
        ctx[gi].d_Vpart,
        (size_t)segN_half*sizeof(float2),
        cudaMemcpyDeviceToHost,
        stream
      ));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    std::vector<float2> hViss_half(segN_half);
    for(int i=0;i<segN_half;i++){
      double re=0, im=0;
      for(int gi=0; gi<G; ++gi){
        re += (double)ctx[gi].h_Vpart[i].x;
        im += (double)ctx[gi].h_Vpart[i].y;
      }
      hViss_half[i]=make_float2((float)re,(float)im);
    }

    phase_correct_viss_halfsym_host(hViss_half.data(), gen.h_w, segN_half);

    std::vector<float2> hViss_seg(segN, make_float2(0,0));
    expand_viss_halfsym_to_full(hViss_half, hViss_seg);

    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].stream;
      CHECK_CUDA(cudaMemcpyAsync(
        ctx[gi].d_Viss,
        hViss_seg.data(),
        (size_t)segN*sizeof(float2),
        cudaMemcpyHostToDevice,
        stream
      ));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    if(write_viss){
      for(int i=0;i<segN;i++){
        ofsViss << hViss_seg[i].x << " " << hViss_seg[i].y << "\n";
      }
    }

    double t_vseg_s = t_vseg.toc_s();

    HostTimer t_rseg;
    t_rseg.tic();

    // ---------- reconstruction params ----------
    float fa=0, fb=0;
    double denom=0;
    compute_fa_fb_host_matlab_no_intercept(gen.h_u, gen.h_v, gen.h_w, segN, fa, fb, denom);

    float umin,umax,vmin,vmax,wmin,wmax;
    if(print_stats) uvw_stats(gen.h_u, gen.h_v, gen.h_w, segN, umin,umax,vmin,vmax,wmin,wmax);

    float lmax=std::sqrt(1.0f+fa*fa);
    float mmax=std::sqrt(1.0f+fb*fb);
    float lmmax=std::max(lmax,mmax);
    float temp_res=std::ceil(2.0f*bl_max/lamda*2.0f*lmmax);
    int RES=(int)(temp_res + 2 + 1 - std::fmod(temp_res,2.0f));
    int half=(RES-1)/2;
    float du=2.0f*bl_max/lamda/(RES-1);

    GroupArtifactsHost grp;
    CHECK_CUDA(cudaSetDevice(ctx[0].dev));
    build_groups_device0(ctx[0].dev, ctx[0].stream,
                         ctx[0].d_u, ctx[0].d_v,
                         ctx[0].d_Viss,
                         segN, du, RES, half,
                         blockage,
                         grp);
    if(grp.nuniq<=0){
      std::cout<<"[seg "<<(s+1)<<"/"<<segs<<"] nuniq<=0, skip\n";
      continue;
    }
    if(blockage==0) W_scalar_total += (long long)grp.nuniq;

    // broadcast group artifacts
    std::vector<int*> d_keys(G,nullptr);
    std::vector<float2*> d_vavg(G,nullptr);
    std::vector<int*> d_offsets(G,nullptr);
    std::vector<int*> d_idx(G,nullptr);

    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].stream;

      CHECK_CUDA(cudaMalloc(&d_keys[gi], (size_t)grp.nuniq*sizeof(int)));
      CHECK_CUDA(cudaMemcpyAsync(d_keys[gi], grp.keys_unique.data(), (size_t)grp.nuniq*sizeof(int), cudaMemcpyHostToDevice, stream));

      if(blockage==0){
        CHECK_CUDA(cudaMalloc(&d_vavg[gi], (size_t)grp.nuniq*sizeof(float2)));
        CHECK_CUDA(cudaMemcpyAsync(d_vavg[gi], grp.viss_avg.data(), (size_t)grp.nuniq*sizeof(float2), cudaMemcpyHostToDevice, stream));
      } else {
        CHECK_CUDA(cudaMalloc(&d_offsets[gi], (size_t)(grp.nuniq+1)*sizeof(int)));
        CHECK_CUDA(cudaMemcpyAsync(d_offsets[gi], grp.offsets.data(), (size_t)(grp.nuniq+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMalloc(&d_idx[gi], (size_t)segN*sizeof(int)));
        CHECK_CUDA(cudaMemcpyAsync(d_idx[gi], grp.idx_sorted.data(), (size_t)segN*sizeof(int), cudaMemcpyHostToDevice, stream));
      }
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // per GPU recon compute and accumulate
    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].stream;

      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cseg, 0, (size_t)ctx[gi].n_chunk*sizeof(float), stream));
      if(blockage==1) CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wseg, 0, (size_t)ctx[gi].n_chunk*sizeof(uint32_t), stream));

      int BLOCK=256;
      int grid=(int)((ctx[gi].n_chunk + BLOCK - 1)/BLOCK);

      if(blockage==0){
        recon_seg_keys_avg<256><<<grid,BLOCK,0,stream>>>(
          ctx[gi].n_chunk,
          ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,
          d_keys[gi], d_vavg[gi],
          grp.nuniq,
          RES,half,du,fa,fb,
          ctx[gi].d_Cseg
        );
      } else {
        recon_seg_blockage<64><<<grid,BLOCK,0,stream>>>(
          ctx[gi].n_chunk,
          ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,
          d_keys[gi], d_offsets[gi], d_idx[gi],
          ctx[gi].d_Viss,
          ctx[gi].d_x1,ctx[gi].d_y1,ctx[gi].d_z1,ctx[gi].d_invn1,
          ctx[gi].d_x2,ctx[gi].d_y2,ctx[gi].d_z2,ctx[gi].d_invn2,
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

    // free group artifacts
    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      if(d_keys[gi]) cudaFree(d_keys[gi]);
      if(d_vavg[gi]) cudaFree(d_vavg[gi]);
      if(d_offsets[gi]) cudaFree(d_offsets[gi]);
      if(d_idx[gi]) cudaFree(d_idx[gi]);
    }

    double t_rseg_s = t_rseg.toc_s();

    std::cout<<"[seg "<<(s+1)<<"/"<<segs<<"] t0="<<t0
             <<" segN="<<segN
             <<" segN_half="<<segN_half
             <<" VissTime="<<t_vseg_s<<"s"
             <<" ReconSegTime="<<t_rseg_s<<"s"
             <<" fa="<<fa<<" fb="<<fb
             <<" denom="<<denom
             <<" nuniq="<<grp.nuniq
             <<" RES="<<RES;
    if(print_stats){
      std::cout<<" u=["<<umin<<","<<umax<<"] v=["<<vmin<<","<<vmax<<"] w=["<<wmin<<","<<wmax<<"]";
    }
    std::cout<<"\n";
  }

    // normalize
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    cudaStream_t stream=ctx[gi].stream;
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

  // write C (concatenate GPU chunks)
  HostTimer t_w;
  t_w.tic();
  std::string outC=out_dir+"C"+std::to_string(day)+"day"+btag+".txt";
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
  std::cout<<"Wrote "<<outC<<" in "<<t_w.toc_s()<<" s\n";

  if(write_viss){
    ofsViss.close();
  }
  if(write_baseline_txt){
    ofsUVW.close(); ofsXYZa.close(); ofsXYZb.close();
  }

  orbit_gen_destroy(gen);

  // cleanup
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);
    if(ctx[gi].h_Vpart) cudaFreeHost(ctx[gi].h_Vpart);

    if(ctx[gi].d_Wseg) cudaFree(ctx[gi].d_Wseg);
    if(ctx[gi].d_Wacc) cudaFree(ctx[gi].d_Wacc);
    if(ctx[gi].d_Cseg) cudaFree(ctx[gi].d_Cseg);
    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);

    if(ctx[gi].d_Viss) cudaFree(ctx[gi].d_Viss);
    if(ctx[gi].d_Vpart) cudaFree(ctx[gi].d_Vpart);

    if(ctx[gi].d_invn2) cudaFree(ctx[gi].d_invn2);
    if(ctx[gi].d_invn1) cudaFree(ctx[gi].d_invn1);
    if(ctx[gi].d_z2) cudaFree(ctx[gi].d_z2);
    if(ctx[gi].d_y2) cudaFree(ctx[gi].d_y2);
    if(ctx[gi].d_x2) cudaFree(ctx[gi].d_x2);
    if(ctx[gi].d_z1) cudaFree(ctx[gi].d_z1);
    if(ctx[gi].d_y1) cudaFree(ctx[gi].d_y1);
    if(ctx[gi].d_x1) cudaFree(ctx[gi].d_x1);

    if(ctx[gi].d_w) cudaFree(ctx[gi].d_w);
    if(ctx[gi].d_v) cudaFree(ctx[gi].d_v);
    if(ctx[gi].d_u) cudaFree(ctx[gi].d_u);

    if(ctx[gi].d_n) cudaFree(ctx[gi].d_n);
    if(ctx[gi].d_m) cudaFree(ctx[gi].d_m);
    if(ctx[gi].d_l) cudaFree(ctx[gi].d_l);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].stream) cudaStreamDestroy(ctx[gi].stream);
  }

  return 0;
}