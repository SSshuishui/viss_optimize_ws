#include "production/common_mirror_tile_vissfast_streamonly.hpp"
#include "allocation_tracking.hpp"
#include "production/orbit_online_mirror_tile_vissfast_streamonly.cuh"
#include "production/viss_recon_kernel_mirror_tile_vissfast_streamonly.cuh"
#include "validated/candidate_generated.cuh"
#include "regression_support.hpp"

int main(int argc,char** argv){
  try {
  HostTimer process_timer;process_timer.tic();
  const std::string stage=get_arg(argc,argv,"--regression_stage","INVALID");
  const int reg_mode=to_int(get_arg(argc,argv,"--regression_mode","-1"),-1);
  const std::string reg_identity=get_arg(argc,argv,"--execution_identity","");
  const std::string refs_dir=get_arg(argc,argv,"--references","");
  prod::require((stage=="prefix"||stage=="capacity"||stage=="full")&&(reg_mode==0||reg_mode==4)&&reg_identity.size()==64&&!refs_dir.empty(),"regression CLI identity/stage");
  prod::pause_path=get_arg(argc,argv,"--pause_file","");
  std::signal(SIGTERM,prod::stop_handler);std::signal(SIGINT,prod::stop_handler);
  prod::References references(refs_dir);
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096);
  int day=to_int(get_arg(argc,argv,"--day","1"),1);
  int day_start=to_int(get_arg(argc,argv,"--day_start",std::to_string(day)), day);
  int day_count=to_int(get_arg(argc,argv,"--day_count","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);
  int segment_limit=to_int(get_arg(argc,argv,"--segment_limit","0"),0);
  int sample_count=to_int(get_arg(argc,argv,"--sample_count","0"),0);
  if(segment_limit<0 || sample_count<0 || sample_count>65536){ std::cerr<<"ERROR short profile controls\n"; return 1; }

  int blockage=to_int(get_arg(argc,argv,"--blockage","1"),1);
  int occ_mode=to_int(get_arg(argc,argv,"--occ_mode","2"),2);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","0"),0);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  std::string C_mode=get_arg(argc,argv,"--C_mode","bin");
  int write_output=to_int(get_arg(argc,argv,"--write_output","1"),1);
  std::string run_tag=get_arg(argc,argv,"--run_tag","");
  std::string summary_csv=get_arg(argc,argv,"--summary_csv","");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 12345ULL);

  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  if(!gpus_s.empty()) setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

  int print_stats=to_int(get_arg(argc,argv,"--print_stats","1"),1);
  int viss_tile_pix=to_int(get_arg(argc,argv,"--viss_tile_pix","256"),256);
  if(viss_tile_pix!=256 && viss_tile_pix!=512){ std::cerr<<"ERROR --viss_tile_pix supports 256 or 512\n"; return 1; }
  long long sky_slab_pix=to_u64(get_arg(argc,argv,"--sky_slab_pix","0"),0);

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

  const std::string uuid2="GPU-d26f44d0-c546-24e1-433c-49a053ecf0ee",uuid3="GPU-99fbd651-f080-aa22-391b-f33caa723578";
  prod::require(gpus_s==(stage=="prefix"?uuid2:uuid2+","+uuid3),"only frozen GPU2/3 UUIDs in fixed order allowed");
  prod::require(btag=="30M"&&nside==16384&&day_start==1&&day_count==1&&segs==10&&segment_limit==1&&sample_count==2048&&blockage==1&&occ_mode==2&&write_output==0&&write_viss==0&&write_baseline_txt==0&&B_mode=="bin"&&orbit_seed==42&&viss_tile_pix==256&&gen_gpu_index==0,"fixed full original production parameters");
  prod::require(sky_slab_pix==(1LL<<25)||sky_slab_pix==(1LL<<27),"frozen IO layouts only");
  prod::require(stage!="prefix"||sky_slab_pix==(1LL<<25),"prefix uses first original 2^25 pixels");
  int requested_G=stage=="prefix"?1:2;
  prodalloc::log.open(out_dir+"allocation.csv");prodalloc::log<<"logical,requested_bytes,rc,free_bytes,total_bytes\n";
  prod::require(!prod::pause(),"PAUSE before CUDA work");

  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  if(devCount <= 0){
    std::cerr<<"ERROR no visible CUDA devices after applying CUDA_VISIBLE_DEVICES="<<gpus_s<<"\n";
    return 1;
  }

  prod::require(devCount==requested_G,"visible CUDA count must equal frozen UUID list; no fallback");
  int G=devCount;

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
           <<" gpus="<<gpus_s<<" C_mode="<<C_mode
           <<" write_output="<<write_output<<" run_tag="<<run_tag<<"\n";

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

  prod::require(npix==3221225472LL&&max_tlen==20944&&max_segN_half==586432&&OrbitRes==62832&&T==209440,"full 30MHz computational plan");
  std::vector<GpuCtx> ctx(G);
  std::vector<std::array<double2*,2>> d_state(G),d_comp(G);
  std::vector<size_t> peak_used_bytes((size_t)G, 0);
  auto sample_gpu_memory = [&](){
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(gpus[gi]));
      size_t free_b=0, total_b=0;
      CHECK_CUDA(cudaMemGetInfo(&free_b, &total_b));
      size_t used_b = total_b - free_b;
      peak_used_bytes[(size_t)gi] = std::max(peak_used_bytes[(size_t)gi], used_b);
    }
  };
  long long chunk_size=(npix + G - 1)/G;
  if(sky_slab_pix<=0){
    std::cerr << "ERROR stream-only version requires --sky_slab_pix>0\n";
    return 1;
  }
  if(sky_slab_pix > chunk_size) sky_slab_pix = chunk_size;
  bool full_sky_resident = false;
  long long slabs_total = 0;
  long long slabs_per_gpu_max = 0;
  for(int gi=0; gi<G; ++gi){
    long long pix0=(long long)gi*chunk_size;
    long long pix1=std::min(npix,pix0+chunk_size);
    long long n_chunk=std::max(0LL,pix1-pix0);
    long long ns = (n_chunk + sky_slab_pix - 1) / sky_slab_pix;
    slabs_total += ns;
    slabs_per_gpu_max = std::max(slabs_per_gpu_max, ns);
  }
  std::cout << "sky_slab_pix=" << sky_slab_pix
            << " slabs_per_gpu_max=" << slabs_per_gpu_max
            << " slabs_total=" << slabs_total
            << " full_sky_resident=" << (full_sky_resident?1:0) << " (stream-only)\n";

  HostTimer t_init;
  t_init.tic();
  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    int dev=gpus[gi];
    CHECK_CUDA(cudaSetDevice(dev));

    long long pix0=(long long)gi*chunk_size;
    long long pix1=std::min(npix,pix0+chunk_size);
    long long n_chunk=std::max(0LL,pix1-pix0);
    long long slab_cap = std::min<long long>(sky_slab_pix, n_chunk);
    long long work_cap = full_sky_resident ? n_chunk : slab_cap;

    ctx[gi].dev=dev;
    ctx[gi].pix0=pix0;
    ctx[gi].pix1=pix1;
    ctx[gi].n_chunk=n_chunk;
    ctx[gi].N=N;
    ctx[gi].N_half=max_segN_half;
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].compute_stream,cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].xfer_stream,cudaStreamNonBlocking));

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)work_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)work_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)work_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)work_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_nm1,(size_t)work_cap*sizeof(float)));
    int ntile_cap = (int)((work_cap + viss_tile_pix - 1) / viss_tile_pix);
    ctx[gi].ntile = ntile_cap;
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx,(size_t)ntile_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy,(size_t)ntile_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz,(size_t)ntile_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA,(size_t)ntile_cap*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA,(size_t)ntile_cap*sizeof(float)));

    if(full_sky_resident){
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
          ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].d_nm1, ctx[gi].n_chunk,
          ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
          ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile);
      }else{
        build_tile_cone_meta_kernel<512><<<ctx[gi].ntile, 256, 0, ctx[gi].xfer_stream>>>(
          ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].d_nm1, ctx[gi].n_chunk,
          ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
          ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ctx[gi].ntile);
      }
      CHECK_CUDA(cudaPeekAtLastError());
    }

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
      CHECK_CUDA(cudaMalloc(&sg.d_Viss,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&d_state[gi][slot],(size_t)max_segN_half*sizeof(double2)));
      CHECK_CUDA(cudaMalloc(&d_comp[gi][slot],(size_t)max_segN_half*sizeof(double2)));
      CHECK_CUDA(cudaMallocHost(&sg.h_Vpart,(size_t)max_segN_half*sizeof(float2)));
      CHECK_CUDA(cudaEventCreateWithFlags(&sg.ready, cudaEventDisableTiming));
    }

    if(stage!="prefix"){
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc,0,(size_t)n_chunk*sizeof(float),ctx[gi].xfer_stream));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cseg,(size_t)work_cap*sizeof(float)));
    if(blockage==1){
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wacc,(size_t)n_chunk*sizeof(uint32_t)));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wacc,0,(size_t)n_chunk*sizeof(uint32_t),ctx[gi].xfer_stream));
      CHECK_CUDA(cudaMalloc(&ctx[gi].d_Wseg,(size_t)work_cap*sizeof(uint32_t)));
    }

    }
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_keys_buf, (size_t)max_segN_half*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_repidx_buf, (size_t)max_segN_half*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_ugu_buf, (size_t)max_segN_half*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_vgu_buf, (size_t)max_segN_half*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_repV_buf, (size_t)max_segN_half*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_repP1_buf, (size_t)max_segN_half*sizeof(float4)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_repP2_buf, (size_t)max_segN_half*sizeof(float4)));

    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)(1<<20)*sizeof(float)));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
  }
  double init_gpu_s = t_init.toc_s();
  sample_gpu_memory();
  std::cout<<"Init GPUs in "<<init_gpu_s<<" s\n";

  BaselineBroadcastInfo binfo;
  if(gen_gpu_index < 0 || gen_gpu_index >= G){
    std::cerr << "ERROR gen_gpu_index out of range: " << gen_gpu_index << " / " << G << " (visible GPUs)\n";
    return 1;
  }
  query_peer_access(binfo, gpus[gen_gpu_index], ctx);
  std::cout << "Baseline broadcast: gen_dev=" << gpus[gen_gpu_index]
            << " need_host_stage=" << binfo.need_host_stage << "\n";

  auto prepare_slab_viss = [&](int gi, long long slab_off, int slabN, cudaStream_t stream)->int{
    if(full_sky_resident) return ctx[gi].ntile;
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B, hB.data() + ctx[gi].pix0 + slab_off, (size_t)slabN*sizeof(float), cudaMemcpyHostToDevice, stream));
    int BLOCK=256;
    int grid=(slabN+BLOCK-1)/BLOCK;
    pix2lmn_nest_kernel<<<grid,BLOCK,0,stream>>>(nside, (unsigned int)(ctx[gi].pix0 + slab_off), slabN, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    int grid_nm1=(slabN+255)/256;
    build_nm1_kernel<<<grid_nm1,256,0,stream>>>(ctx[gi].d_n, ctx[gi].d_nm1, slabN);
    int ntile_cur = (int)((slabN + viss_tile_pix - 1) / viss_tile_pix);
    if(viss_tile_pix==256){
      build_tile_cone_meta_kernel<256><<<ntile_cur, 256, 0, stream>>>(
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].d_nm1, slabN,
        ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
        ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ntile_cur);
    }else{
      build_tile_cone_meta_kernel<512><<<ntile_cur, 256, 0, stream>>>(
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].d_nm1, slabN,
        ctx[gi].d_tile_cx, ctx[gi].d_tile_cy, ctx[gi].d_tile_cz,
        ctx[gi].d_tile_cosA, ctx[gi].d_tile_sinA, ntile_cur);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    return ntile_cur;
  };

  auto prepare_slab_recon = [&](int gi, long long slab_off, int slabN, cudaStream_t stream){
    if(full_sky_resident) return;
    int BLOCK=256;
    int grid=(slabN+BLOCK-1)/BLOCK;
    pix2lmn_nest_kernel<<<grid,BLOCK,0,stream>>>(nside, (unsigned int)(ctx[gi].pix0 + slab_off), slabN, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    CHECK_CUDA(cudaPeekAtLastError());
  };

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
    HostTimer t_day_total; t_day_total.tic();
    double visibility_total_s = 0.0;
    double reconstruction_total_s = 0.0;
    long long occupied_groups_total = 0;
    std::cout << "===== Day " << dayid << " / " << (day_start + day_count - 1) << " =====\n";
    if(stage!="prefix") reset_daily_accumulators();

    std::ofstream ofsViss;
    if(write_viss){
      std::string fv=out_dir+"Viss"+std::to_string(dayid)+"day"+btag+".txt";
      ofsViss.open(fv);
      if(!ofsViss.is_open()){ std::cerr << "ERROR open " << fv << "\n"; return 1; }
      static thread_local std::vector<char> vissbuf(8<<20);
      ofsViss.rdbuf()->pubsetbuf(vissbuf.data(), vissbuf.size());
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
    }

    OrbitGenCtx gen;
    HostTimer t_gen_init; t_gen_init.tic();
    orbit_gen_init(gen, gpus[gen_gpu_index], lambda_m, dayid, max_tlen, orbit_seed);
    std::cout << "Orbit generator initialized on dev=" << gen.dev << " in " << t_gen_init.toc_s() << " s\n";

    bool need_all_host = true;
    orbit_gen_make_segment_async(gen, 0, seg_t0[0], seg_tlen[0], need_all_host);
    broadcast_segment_to_gpus_async(gen, 0, binfo, ctx, need_all_host);
    if(segs > 1) orbit_gen_make_segment_async(gen, 1, seg_t0[1], seg_tlen[1], need_all_host);

    HostTimer t_rec; t_rec.tic();
    long long W_scalar_total=0;

    for(int s=0;s<segs && (segment_limit==0 || s<segment_limit);s++){
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

      uint64_t geom_hash=references.geometry(gcur,cosphi);
      prod::require(t0==0&&tlen==20944&&segN_half==586432,"first original segment/full half baseline identity");
      prod::memory(out_dir,G,stage!="prefix");
      HostTimer t_vseg; t_vseg.tic();
      const int BLOCKV=256;
      int gridV=(segN_half + BLOCKV - 1)/BLOCKV;
      size_t shmem=(size_t)viss_tile_pix*5*sizeof(float);
      std::vector<std::vector<float2>> partial_gpu(G, std::vector<float2>((size_t)segN_half, make_float2(0,0)));


      if(stage=="prefix"){
        int gi=0;CHECK_CUDA(cudaSetDevice(0));auto stream=ctx[0].compute_stream;auto&slot=ctx[0].slots[cur];
        CHECK_CUDA(cudaStreamWaitEvent(stream,slot.ready,0));int slabN=1<<25;
        int ntile_cur=prepare_slab_viss(0,0,slabN,stream);
        HostTimer prefix_original;prefix_original.tic();
        viss_partial_all_halfsym_tilecone<256,true><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart);
        std::vector<float2>a(segN_half),b(segN_half);CHECK_CUDA(cudaMemcpyAsync(a.data(),slot.d_Vpart,a.size()*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        double prefix_original_s=prefix_original.toc_s();
        slab_candidate<256,true,0><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart,d_state[0][cur],d_comp[0][cur]);
        CHECK_CUDA(cudaMemcpyAsync(b.data(),slot.d_Vpart,b.size()*8,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        bool clone=!std::memcmp(a.data(),b.data(),a.size()*8);prod::text(std::filesystem::path(out_dir)/"clone_gate.json",std::string("{\"pass\":")+(clone?"true":"false")+",\"bitwise\":"+(clone?"true":"false")+",\"all_half_baselines\":586432,\"input_pixels\":33554432}\n");prod::require(clone,"original header versus MODE0 clone");
        for(int z=0;z<2;z++){CHECK_CUDA(cudaMemsetAsync(d_state[0][z],0,segN_half*16,stream));CHECK_CUDA(cudaMemsetAsync(d_comp[0][z],0,segN_half*16,stream));}
        HostTimer prefix_micro;prefix_micro.tic();
        slab_candidate<256,true,4><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart,d_state[0][cur],d_comp[0][cur]);
        std::vector<double2>state(segN_half);CHECK_CUDA(cudaMemcpyAsync(state.data(),d_state[0][cur],state.size()*16,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));
        double prefix_micro_s=prefix_micro.toc_s();
        references.state(state,std::filesystem::path(out_dir)/"prefix_micro_reference",true);
        // Exercise reset of the alternate state buffer using the same first-segment geometry.
        CHECK_CUDA(cudaMemsetAsync(d_state[0][cur^1],0x5a,segN_half*16,stream));CHECK_CUDA(cudaMemsetAsync(d_state[0][cur^1],0,segN_half*16,stream));
        slab_candidate<256,true,4><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart,d_state[0][cur^1],d_comp[0][cur^1]);
        std::vector<double2>second(segN_half);CHECK_CUDA(cudaMemcpyAsync(second.data(),d_state[0][cur^1],second.size()*16,cudaMemcpyDeviceToHost,stream));CHECK_CUDA(cudaStreamSynchronize(stream));prod::require(!std::memcmp(state.data(),second.data(),state.size()*16),"alternate-buffer reset gate");
        prod::text(std::filesystem::path(out_dir)/"prefix.json","{\"pass\":true,\"all_half_baselines\":586432,\"input_pixels\":33554432,\"original_plan_segments\":10,\"reset_buffers_bitwise\":true,\"scope\":\"one first-segment prefix and alternate state-buffer reset; no full C/W allocated; not complete production\"}\n");
        std::ostringstream costs;costs<<std::setprecision(12)<<"{\"original_first33M_forward_s\":"<<prefix_original_s<<",\"micro_first33M_forward_s\":"<<prefix_micro_s<<",\"includes_D2H_sync\":true,\"total_internal_process_s\":"<<process_timer.toc_s()<<"}\n";prod::text(std::filesystem::path(out_dir)/"pilot_cost.json",costs.str());
        std::cout<<"PREFIX_GATE_PASS elapsed_s="<<process_timer.toc_s()<<std::endl;return 0;
      }

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
          if(reg_mode==0){viss_partial_all_halfsym_tilecone<256,true><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart);}else{slab_candidate<256,true,4><<<gridV,BLOCKV,shmem,stream>>>(
            ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,slabN,
            ctx[gi].d_tile_cx,ctx[gi].d_tile_cy,ctx[gi].d_tile_cz,ctx[gi].d_tile_cosA,ctx[gi].d_tile_sinA,ntile_cur,
            slot.d_u,slot.d_v,slot.d_w,slot.d_x1,slot.d_y1,slot.d_z1,slot.d_invn1,
            slot.d_x2,slot.d_y2,slot.d_z2,slot.d_invn2,segN_half,cosphi,slot.d_Vpart,d_state[gi][cur],d_comp[gi][cur]);}
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

      std::vector<float2> hViss_half(segN_half, make_float2(0,0));
      for(int i=0;i<segN_half;i++){
        for(int gi=0; gi<G; ++gi){
          hViss_half[i].x += partial_gpu[gi][i].x;
          hViss_half[i].y += partial_gpu[gi][i].y;
        }
      }
      if(stage!="capacity"){
        if(cached){hViss_half=prod::load_forward(out_dir,prod::forward_identity(reg_identity,reg_mode,G,sky_slab_pix,geom_hash));prod::checkpoint_cleanup(out_dir,G);}
        else{prod::commit_forward(out_dir,prod::forward_identity(reg_identity,reg_mode,G,sky_slab_pix,geom_hash),hViss_half);prod::checkpoint_cleanup(out_dir,G);}
        for(const auto&v:hViss_half)prod::require(std::isfinite(v.x)&&std::isfinite(v.y),"full forward V finite gate");
      }
      phase_correct_viss_halfsym_host(hViss_half.data(), gcur.h_w, segN_half);
      if(stage!="capacity")prod::blob(std::filesystem::path(out_dir)/"forward_consumed.f32x2",hViss_half.data(),hViss_half.size()*8);

      std::vector<float2> hViss_seg;
      if(write_viss){ hViss_seg.assign(segN, make_float2(0,0)); expand_viss_halfsym_to_full(hViss_half, hViss_seg); }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaMemcpyAsync(slot.d_Viss, hViss_half.data(), (size_t)segN_half*sizeof(float2), cudaMemcpyHostToDevice, stream));
      }
      if(write_viss){ for(int i=0;i<segN;i++) ofsViss << hViss_seg[i].x << " " << hViss_seg[i].y << "\n"; }
      double t_vseg_s = t_vseg.toc_s();
      visibility_total_s += t_vseg_s;

      HostTimer t_rseg; t_rseg.tic();
      float fa=0, fb=0; double denom=0;
      compute_fa_fb_host_matlab_no_intercept(gcur.h_u, gcur.h_v, gcur.h_w, segN, fa, fb, denom);
      float umin,umax,vmin,vmax,wmin,wmax;
      if(print_stats) uvw_stats(gcur.h_u, gcur.h_v, gcur.h_w, segN, umin,umax,vmin,vmax,wmin,wmax);
      float lmax=std::sqrt(1.0f+fa*fa), mmax=std::sqrt(1.0f+fb*fb), lmmax=std::max(lmax,mmax);
      float temp_res=std::ceil(2.0f*bl_max/lamda*2.0f*lmmax);
      int RES=(int)(temp_res + 2 + 1 - std::fmod(temp_res,2.0f));
      int half=(RES-1)/2;
      float du=2.0f*bl_max/lamda/(RES-1);

      GroupArtifactsHost grp;
      CHECK_CUDA(cudaSetDevice(ctx[0].dev));
      build_groups_device0_half(ctx[0].dev, ctx[0].compute_stream, ctx[0].slots[cur].d_u, ctx[0].slots[cur].d_v, ctx[0].slots[cur].d_Viss, segN_half, du, RES, half, blockage, occ_mode, grp);
      if(grp.nuniq<=0){ std::cout<<"[day "<<dayid<<" seg "<<(s+1)<<"/"<<segs<<"] nuniq<=0, skip\n"; continue; }
      references.groups(grp,fa,fb,du,RES,half);
      prod::memory(out_dir,G,true);
      prod::text(std::filesystem::path(out_dir)/"group_gate.json","{\"pass\":true,\"half_baselines\":586432,\"groups\":86668,\"full_keys_repidx_geometry_fit_bitwise\":true}\n");
      if(stage=="capacity"){prod::text(std::filesystem::path(out_dir)/"capacity.json","{\"pass\":true,\"G\":2,\"npix\":3221225472,\"resident_C_W_bytes_each\":12884901888,\"full_geometry_slots\":2,\"original_full_CUB_group_executed\":true}\n");return 0;}
      occupied_groups_total += (long long)grp.nuniq;
      if(blockage==0) W_scalar_total += (long long)grp.nuniq;

      if(s+1 < segs){ orbit_gen_wait_slot(gen, nxt); broadcast_segment_to_gpus_async(gen, nxt, binfo, ctx, need_all_host); }
      if(s+2 < segs){ orbit_gen_make_segment_async(gen, cur, seg_t0[s+2], seg_tlen[s+2], need_all_host); }

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
          host_ugu[gi].resize(grp.nuniq); host_vgu[gi].resize(grp.nuniq);
          for(int q=0; q<grp.nuniq; ++q){
            int key = grp.keys_unique[q]; int tmp = key - 1; int U = tmp / RES; int V = tmp - U * RES; int ui = U - half; int vi = V - half;
            host_ugu[gi][q] = ui * du; host_vgu[gi][q] = vi * du;
          }
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_repidx_buf, grp.rep_half_idx.data(), (size_t)grp.nuniq*sizeof(int), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_ugu_buf, host_ugu[gi].data(), (size_t)grp.nuniq*sizeof(float), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_vgu_buf, host_vgu[gi].data(), (size_t)grp.nuniq*sizeof(float), cudaMemcpyHostToDevice, stream));
          int b0=256, g0=(grp.nuniq + b0 - 1)/b0;
          gather_rep_half_inputs<<<g0,b0,0,stream>>>(ctx[gi].d_repidx_buf, grp.nuniq, ctx[gi].slots[cur].d_Viss,
            ctx[gi].slots[cur].d_x1, ctx[gi].slots[cur].d_y1, ctx[gi].slots[cur].d_z1, ctx[gi].slots[cur].d_invn1,
            ctx[gi].slots[cur].d_x2, ctx[gi].slots[cur].d_y2, ctx[gi].slots[cur].d_z2, ctx[gi].slots[cur].d_invn2,
            ctx[gi].d_repV_buf, ctx[gi].d_repP1_buf, ctx[gi].d_repP2_buf);
        } else {
          CHECK_CUDA(cudaMalloc(&d_offsets[gi], (size_t)(grp.nuniq+1)*sizeof(int)));
          CHECK_CUDA(cudaMemcpyAsync(d_offsets[gi], grp.offsets.data(), (size_t)(grp.nuniq+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
          CHECK_CUDA(cudaMalloc(&d_idx[gi], (size_t)segN_half*sizeof(int)));
          CHECK_CUDA(cudaMemcpyAsync(d_idx[gi], grp.idx_sorted.data(), (size_t)segN_half*sizeof(int), cudaMemcpyHostToDevice, stream));
        }
      }

      sample_gpu_memory();
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        long long work_cap = full_sky_resident ? ctx[gi].n_chunk : sky_slab_pix;
        for(long long slab_off=0; slab_off<ctx[gi].n_chunk; slab_off += sky_slab_pix){
          if(prod::pause()){CHECK_CUDA(cudaStreamSynchronize(stream));break;}
          int slabN = (int)std::min<long long>(sky_slab_pix, ctx[gi].n_chunk - slab_off);
          if(!full_sky_resident) prepare_slab_recon(gi, slab_off, slabN, stream);
          CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cseg, 0, (size_t)slabN*sizeof(float), stream));
          if(blockage==1) CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Wseg, 0, (size_t)slabN*sizeof(uint32_t), stream));
          int BLOCK=256; int grid=(int)((slabN + BLOCK - 1)/BLOCK);
          if(blockage==0){
            recon_seg_keys_avg_half<256><<<grid,BLOCK,0,stream>>>(slabN, ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n, d_keys[gi], d_vavg[gi], grp.nuniq, RES,half,du,fa,fb, ctx[gi].d_Cseg);
          } else if(occ_mode==2 && !grp.rep_half_idx.empty()) {
            recon_seg_blockage_half_rep<64><<<grid,BLOCK,0,stream>>>(slabN, ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n, d_keys[gi], ctx[gi].d_ugu_buf, ctx[gi].d_vgu_buf, ctx[gi].d_repV_buf, ctx[gi].d_repP1_buf, ctx[gi].d_repP2_buf, grp.nuniq, fa,fb, cosphi, ctx[gi].d_Cseg, ctx[gi].d_Wseg);
          } else {
            recon_seg_blockage_half<64><<<grid,BLOCK,0,stream>>>(slabN, ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n, d_keys[gi], d_offsets[gi], d_idx[gi], ctx[gi].slots[cur].d_Viss, ctx[gi].slots[cur].d_x1,ctx[gi].slots[cur].d_y1,ctx[gi].slots[cur].d_z1,ctx[gi].slots[cur].d_invn1, ctx[gi].slots[cur].d_x2,ctx[gi].slots[cur].d_y2,ctx[gi].slots[cur].d_z2,ctx[gi].slots[cur].d_invn2, grp.nuniq, RES,half,du,fa,fb, cosphi, occ_mode, ctx[gi].d_Cseg, ctx[gi].d_Wseg);
          }
          CHECK_CUDA(cudaPeekAtLastError());
          int b=256; int g2=(int)((slabN+b-1)/b);
          add_inplace<<<g2,b,0,stream>>>(ctx[gi].d_Cacc + slab_off, ctx[gi].d_Cseg, slabN);
          if(blockage==1) add_inplace_u32<<<g2,b,0,stream>>>(ctx[gi].d_Wacc + slab_off, ctx[gi].d_Wseg, slabN);
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
      }

      if(prod::pause())throw std::runtime_error("paused inverse; explicit resume replays inverse from committed full forward V");
      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        if(d_offsets[gi]) CHECK_CUDA(cudaFree(d_offsets[gi]));
        if(d_idx[gi]) CHECK_CUDA(cudaFree(d_idx[gi]));
      }
      sample_gpu_memory();
      double t_rseg_s = t_rseg.toc_s();
      reconstruction_total_s += t_rseg_s;
      std::cout<<"[day "<<dayid<<" seg "<<(s+1)<<"/"<<segs<<"] t0="<<t0
               <<" segN="<<segN<<" segN_half="<<segN_half
               <<" VissTime="<<t_vseg_s<<"s"
               <<" ReconSegTime="<<t_rseg_s<<"s"
               <<" fa="<<fa<<" fb="<<fb<<" denom="<<denom<<" nuniq_halfgrid="<<grp.nuniq<<" RES="<<RES;
      if(print_stats) std::cout<<" u=["<<umin<<","<<umax<<"] v=["<<vmin<<","<<vmax<<"] w=["<<wmin<<","<<wmax<<"]";
      std::cout<<"\n";
    }

    prod::scan(ctx,out_dir,"N");prod::samples(ctx,npix,chunk_size,out_dir,"N");
    prod::scan(ctx,out_dir,"W");prod::samples(ctx,npix,chunk_size,out_dir,"W");
    #pragma omp parallel for num_threads(G)
    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream=ctx[gi].compute_stream; int b=256; int g2=(int)((ctx[gi].n_chunk+b-1)/b);
      if(blockage==0){ float inv = (W_scalar_total>0)? (1.0f/(float)W_scalar_total) : 0.0f; scale_inplace<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].n_chunk, inv); }
      else { normalize_by_weight<<<g2,b,0,stream>>>(ctx[gi].d_Cacc, ctx[gi].d_Wacc, ctx[gi].n_chunk); }
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    prod::scan(ctx,out_dir,"C");prod::samples(ctx,npix,chunk_size,out_dir,"C");
    prod::final_info(out_dir,reg_mode,G,sky_slab_pix,process_timer.toc_s());
    double segment_loop_total_s = t_rec.toc_s();
    std::cout<<"SHORT_PROFILE segment_limit="<<segment_limit<<" original_segs="<<segs<<" sample_count="<<sample_count<<"\n";
    if(sample_count>0){
      std::ofstream sample(out_dir+"samples_"+run_tag+".csv");
      if(!sample){ std::cerr<<"ERROR sample CSV\n"; return 1; }
      sample<<"pixel_id,value\n"<<std::setprecision(9);
      for(int k=0;k<sample_count;k++){
        long long pixel=((2LL*k+1)*npix)/(2LL*sample_count);
        int gi=(int)std::min<long long>(pixel/chunk_size,G-1);
        long long local=pixel-ctx[gi].pix0;
        if(pixel<0 || pixel>=npix || local<0 || local>=ctx[gi].n_chunk){
          std::cerr<<"ERROR sampled output index out of range\n"; return 1;
        }
        float val=0;
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        CHECK_CUDA(cudaMemcpy(&val,ctx[gi].d_Cacc+local,sizeof(float),cudaMemcpyDeviceToHost));
        sample<<pixel<<','<<val<<"\n";
      }
    }

    std::cout<<"Segment loop done in "<<segment_loop_total_s<<" s\n";
    double output_write_s = 0.0;
    std::string tag_suffix = run_tag.empty() ? "" : ("_" + run_tag);
    std::string outC = out_dir+"C"+std::to_string(dayid)+"day"+btag+tag_suffix + ((C_mode=="txt")? ".txt" : ".bin");
    if(write_output){
      HostTimer t_w; t_w.tic();
      if(C_mode=="txt"){
        std::ofstream ofs(outC); if(!ofs.is_open()){ std::cerr<<"ERROR open "<<outC<<"\n"; return 1; }
        const int CH=1<<20;
        for(int gi=0; gi<G; ++gi){ CHECK_CUDA(cudaSetDevice(ctx[gi].dev)); for(long long off=0; off<ctx[gi].n_chunk; off+=CH){ int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off); CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost)); for(int i=0;i<cur;i++) ofs<<ctx[gi].h_chunk[i]<<"\n"; } }
      } else {
        std::ofstream ofs(outC, std::ios::binary); if(!ofs.is_open()){ std::cerr<<"ERROR open "<<outC<<"\n"; return 1; }
        const int CH=1<<20;
        for(int gi=0; gi<G; ++gi){ CHECK_CUDA(cudaSetDevice(ctx[gi].dev)); for(long long off=0; off<ctx[gi].n_chunk; off+=CH){ int cur=(int)std::min((long long)CH, ctx[gi].n_chunk-off); CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost)); ofs.write(reinterpret_cast<const char*>(ctx[gi].h_chunk), (std::streamsize)cur*sizeof(float)); } }
      }
      output_write_s = t_w.toc_s();
      std::cout<<"Wrote "<<outC<<" in "<<output_write_s<<" s\n";
    } else {
      std::cout<<"Output writing disabled (--write_output=0)\n";
    }
    sample_gpu_memory();
    size_t peak_max_b = 0;
    for(size_t v: peak_used_bytes) peak_max_b = std::max(peak_max_b, v);
    double peak_max_gib = (double)peak_max_b / (1024.0*1024.0*1024.0);
    double total_day_s = t_day_total.toc_s();
    std::cout<<"DAY_SUMMARY day="<<dayid
             <<" visibility_s="<<visibility_total_s
             <<" reconstruction_s="<<reconstruction_total_s
             <<" segment_loop_s="<<segment_loop_total_s
             <<" output_write_s="<<output_write_s
             <<" total_day_s="<<total_day_s
             <<" slab_pix="<<sky_slab_pix
             <<" slabs_per_gpu_max="<<slabs_per_gpu_max
             <<" slabs_total="<<slabs_total
             <<" peak_gpu_mem_gib="<<peak_max_gib
             <<" occupied_groups_total="<<occupied_groups_total<<"\n";
    if(!summary_csv.empty()){
      bool need_header = true;
      { std::ifstream chk(summary_csv); need_header = !chk.good() || chk.peek()==std::ifstream::traits_type::eof(); }
      std::ofstream csv(summary_csv, std::ios::app);
      if(!csv.is_open()){ std::cerr<<"ERROR open summary csv "<<summary_csv<<"\n"; return 1; }
      if(need_header){
        csv << "run_tag,day,btag,nside,gpus,segs,slab_pix,slabs_per_gpu_max,slabs_total,visibility_s,reconstruction_s,segment_loop_s,output_write_s,total_day_s,peak_gpu_mem_gib,occupied_groups_total,write_output\n";
      }
      csv << run_tag << ',' << dayid << ',' << btag << ',' << nside << ",\"" << gpus_s << "\"," << segs << ','
          << sky_slab_pix << ',' << slabs_per_gpu_max << ',' << slabs_total << ','
          << std::setprecision(10) << visibility_total_s << ',' << reconstruction_total_s << ','
          << segment_loop_total_s << ',' << output_write_s << ',' << total_day_s << ','
          << peak_max_gib << ',' << occupied_groups_total << ',' << write_output << "\n";
    }
    if(write_viss) ofsViss.close();
    if(write_baseline_txt){ ofsUVW.close(); ofsXYZa.close(); ofsXYZb.close(); }
    orbit_gen_destroy(gen);
  }

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);
    if(ctx[gi].d_Wseg) cudaFree(ctx[gi].d_Wseg);
    if(ctx[gi].d_Wacc) cudaFree(ctx[gi].d_Wacc);
    if(ctx[gi].d_Cseg) cudaFree(ctx[gi].d_Cseg);
    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);
    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      if(d_state[gi][slot])cudaFree(d_state[gi][slot]);if(d_comp[gi][slot])cudaFree(d_comp[gi][slot]);
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
    if(ctx[gi].d_repP2_buf) cudaFree(ctx[gi].d_repP2_buf);
    if(ctx[gi].d_repP1_buf) cudaFree(ctx[gi].d_repP1_buf);
    if(ctx[gi].d_repV_buf) cudaFree(ctx[gi].d_repV_buf);
    if(ctx[gi].d_vgu_buf) cudaFree(ctx[gi].d_vgu_buf);
    if(ctx[gi].d_ugu_buf) cudaFree(ctx[gi].d_ugu_buf);
    if(ctx[gi].d_repidx_buf) cudaFree(ctx[gi].d_repidx_buf);
    if(ctx[gi].d_keys_buf) cudaFree(ctx[gi].d_keys_buf);
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
  }catch(const std::exception&e){std::cerr<<"REGRESSION_STOP "<<e.what()<<std::endl;return prod::pause()?75:1;}
}
