#include "common.hpp"
#include "orbit_online.cuh"
#include "viss_recon_kernel.cuh"

#include <array>
#include <cub/iterator/counting_input_iterator.cuh>
#include <limits>

static constexpr int RECON_TILE_PIX_HOST = 256;
static constexpr int RECON_TILE_BL_HOST  = 128;


struct ReconBalanceBlock {
  int owner_gpu = -1;
  long long global_pix0 = 0;
  int len = 0;
  long long owner_off = 0;
};

int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096);
  int day_start=to_int(get_arg(argc,argv,"--day_start","1"),1);
  int day_count=to_int(get_arg(argc,argv,"--day_count","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);
  int dcf_days=to_int(get_arg(argc,argv,"--dcf_days","1"),1);
  std::string dcf_bin=get_arg(argc,argv,"--dcf_bin","");

  int viss_tile_pix=to_int(get_arg(argc,argv,"--viss_tile_pix","256"),256);
  if(viss_tile_pix!=256 && viss_tile_pix!=512){ std::cerr<<"ERROR --viss_tile_pix supports 256 or 512\n"; return 1; }
  int recon_balance_block_tiles=to_int(get_arg(argc,argv,"--recon_balance_block_tiles","256"),256);
  if(recon_balance_block_tiles<=0){ std::cerr<<"ERROR --recon_balance_block_tiles must be > 0\n"; return 1; }
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","0"),0);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  std::string C_mode=get_arg(argc,argv,"--C_mode","bin");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  int reducer_gpu_index=to_int(get_arg(argc,argv,"--reducer_gpu_index","-1"),-1);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","42"), 12345ULL);
  std::string sky_order = get_arg(argc, argv, "--sky_order", "ring");

  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out_direct3d/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  if(!gpus_s.empty()) setenv("CUDA_VISIBLE_DEVICES", gpus_s.c_str(), 1);

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
  for(int i=0;i<G;i++){ std::cout<<gpus[i]; if(i+1<G) std::cout<<","; }
  std::cout<<"\n";

  long long npix=12LL*(long long)nside*(long long)nside;
  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);

  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float bl_max=100e3f;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix
           <<" day_start="<<day_start<<" day_count="<<day_count<<" segs="<<segs
           <<" dcf_days="<<dcf_days<<"\n";
  std::cout<<"blockage=always_on  [Unified forward/adjoint tile operator: Stage-1=forward, Stage-2=adjoint]\n";
  std::cout<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<" C_mode="<<C_mode
           <<" viss_tile_pix="<<viss_tile_pix<<"\n";
  std::cout<<"stage2_load_balance=block_cyclic_pixels recon_balance_block_tiles="<<recon_balance_block_tiles
           <<" recon_balance_block_pix="<<((long long)recon_balance_block_tiles * RECON_TILE_PIX_HOST)<<"\n";
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

  // C=single(B.*s);  
  const double pix_area = 4.0 * M_PI / (double)npix;
  #pragma omp parallel for
  for (long long i = 0; i < npix; ++i) {
    hB[i] = (float)((double)hB[i] * pix_area);
  }
  std::cout << " pix_area=" << pix_area
            << " inv_pix_area=" << (1.0 / pix_area)
            << "\n";

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

  HostTimer t_dcf;
  t_dcf.tic();
  std::vector<long long> mb;
  std::vector<float> hdcf;
  if(gen_gpu_index < 0 || gen_gpu_index >= G){
    std::cerr << "ERROR --gen_gpu_index out of range\n";
    return 1;
  }
  if(reducer_gpu_index < 0) reducer_gpu_index = gen_gpu_index;
  if(reducer_gpu_index < 0 || reducer_gpu_index >= G){
    std::cerr << "ERROR --reducer_gpu_index out of range\n";
    return 1;
  }
  if(!dcf_bin.empty()){
    DcfMbBinPayload payload;
    if(!load_dcf_mb_bin(dcf_bin, payload)){
      std::cerr << "ERROR loading dcf bin: " << dcf_bin << "\n";
      return 1;
    }
    mb = payload.mb;
    hdcf = payload.dcf;
    if(hdcf.empty()){
      std::cerr << "ERROR dcf bin has empty dcf array: " << dcf_bin << "\n";
      return 1;
    }
    std::cout << "Loaded dcf bin from " << dcf_bin
              << " in " << t_dcf.toc_s() << " s, dcf_len=" << hdcf.size()
              << ", mb_len=" << mb.size()
              << ", src_btag=" << payload.btag
              << ", src_days=" << payload.dcf_days
              << ", src_seed=" << payload.orbit_seed << "\n";
    if(payload.btag != btag){
      std::cout << "WARNING dcf bin btag=" << payload.btag << " but current btag=" << btag << "\n";
    }
  } else {
    if(!accumulate_mb_from_online_orbit(gpus[gen_gpu_index], lambda_m, bl_max, dcf_days, segs, orbit_seed, mb)){
      return 1;
    }
    compute_dcf_from_mb(mb, hdcf);
    std::cout << "Computed dcf in " << t_dcf.toc_s() << " s, dcf_len=" << hdcf.size() << "\n";
  }

  std::vector<GpuCtx> ctx(G);
  std::vector<GpuDirectExtra> ext(G);
  long long chunk_size=(npix + G - 1)/G;

  const long long recon_balance_block_pix = (long long)recon_balance_block_tiles * RECON_TILE_PIX_HOST;
  std::vector<ReconBalanceBlock> recon_blocks;
  std::vector<std::vector<ReconBalanceBlock>> recon_blocks_by_gpu(G);
  std::vector<long long> recon_chunk_sizes(G, 0);
  {
    int blk = 0;
    for(long long pix = 0; pix < npix; pix += recon_balance_block_pix, ++blk){
      int owner = blk % G;
      int len = (int)std::min(recon_balance_block_pix, npix - pix);
      ReconBalanceBlock rb;
      rb.owner_gpu = owner;
      rb.global_pix0 = pix;
      rb.len = len;
      rb.owner_off = recon_chunk_sizes[owner];
      recon_chunk_sizes[owner] += (long long)len;
      recon_blocks.push_back(rb);
      recon_blocks_by_gpu[owner].push_back(rb);
    }
  }

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
    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].reduce_stream,cudaStreamNonBlocking));

    long long recon_n_chunk = recon_chunk_sizes[gi];
    ctx[gi].recon_n_chunk = recon_n_chunk;

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_nm1,(size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l_recon,(size_t)recon_n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m_recon,(size_t)recon_n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n_recon,(size_t)recon_n_chunk*sizeof(float)));
    ctx[gi].ntile_viss = (int)((n_chunk + viss_tile_pix - 1) / viss_tile_pix);
    ctx[gi].ntile_recon = (int)((recon_n_chunk + RECON_TILE_PIX_HOST - 1) / RECON_TILE_PIX_HOST);
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA_viss,(size_t)ctx[gi].ntile_viss*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cx_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cy_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cz_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_cosA_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_tile_sinA_recon,(size_t)ctx[gi].ntile_recon*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,hB.data()+pix0,(size_t)n_chunk*sizeof(float),cudaMemcpyHostToDevice,ctx[gi].xfer_stream));

    int BLOCK=256;
    int grid=(int)((n_chunk+BLOCK-1)/BLOCK);
    if (sky_order == "ring") {
      pix2lmn_ring_kernel<<<grid, BLOCK, 0, ctx[gi].xfer_stream>>>(
        nside, (unsigned int)pix0, (int)n_chunk,
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    } else {
      pix2lmn_nest_kernel<<<grid, BLOCK, 0, ctx[gi].xfer_stream>>>(
        nside, (unsigned int)pix0, (int)n_chunk,
        ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n);
    }
    CHECK_CUDA(cudaPeekAtLastError());
    int grid_nm1=(int)((n_chunk+255)/256);
    build_nm1_kernel<<<grid_nm1,256,0,ctx[gi].xfer_stream>>>(ctx[gi].d_n, ctx[gi].d_nm1, ctx[gi].n_chunk);    launch_build_tile_cone_meta_runtime(viss_tile_pix, ctx[gi].xfer_stream,
      ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
      ctx[gi].d_tile_cx_viss, ctx[gi].d_tile_cy_viss, ctx[gi].d_tile_cz_viss,
      ctx[gi].d_tile_cosA_viss, ctx[gi].d_tile_sinA_viss, ctx[gi].ntile_viss);
    for(const auto &rb : recon_blocks_by_gpu[gi]){
      int grid_recon=(int)((rb.len+BLOCK-1)/BLOCK);
      if (sky_order == "ring") {
        pix2lmn_ring_kernel<<<grid_recon, BLOCK, 0, ctx[gi].xfer_stream>>>(
          nside, (unsigned int)rb.global_pix0, rb.len,
          ctx[gi].d_l_recon + rb.owner_off,
          ctx[gi].d_m_recon + rb.owner_off,
          ctx[gi].d_n_recon + rb.owner_off);
      } else {
        pix2lmn_nest_kernel<<<grid_recon, BLOCK, 0, ctx[gi].xfer_stream>>>(
          nside, (unsigned int)rb.global_pix0, rb.len,
          ctx[gi].d_l_recon + rb.owner_off,
          ctx[gi].d_m_recon + rb.owner_off,
          ctx[gi].d_n_recon + rb.owner_off);
      }
      CHECK_CUDA(cudaPeekAtLastError());
    }
    launch_build_tile_cone_meta_runtime(RECON_TILE_PIX_HOST, ctx[gi].xfer_stream,
      ctx[gi].d_l_recon, ctx[gi].d_m_recon, ctx[gi].d_n_recon, ctx[gi].recon_n_chunk,
      ctx[gi].d_tile_cx_recon, ctx[gi].d_tile_cy_recon, ctx[gi].d_tile_cz_recon,
      ctx[gi].d_tile_cosA_recon, ctx[gi].d_tile_sinA_recon, ctx[gi].ntile_recon);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaMalloc(&ext[gi].d_dcf,(size_t)hdcf.size()*sizeof(float)));
    CHECK_CUDA(cudaMemcpyAsync(ext[gi].d_dcf, hdcf.data(), (size_t)hdcf.size()*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].xfer_stream));

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      CHECK_CUDA(cudaMalloc(&sg.d_u, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_v, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_w, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x1, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y1, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z1, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_x2, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_y2, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_z2, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn1, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_invn2, (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMalloc(&sg.d_Vpart, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&sg.d_Viss, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaMalloc(&ext[gi].d_pairw_half[slot], (size_t)max_segN_half * sizeof(float)));
      CHECK_CUDA(cudaMallocHost(&sg.h_Vpart, (size_t)max_segN_half * sizeof(float2)));
      CHECK_CUDA(cudaEventCreate(&sg.ready));
      CHECK_CUDA(cudaEventCreate(&sg.viss_stop));
      CHECK_CUDA(cudaEventCreateWithFlags(&sg.collect_ready, cudaEventDisableTiming));
      CHECK_CUDA(cudaEventCreate(&sg.reduce_done));
      CHECK_CUDA(cudaEventCreate(&sg.scatter_done));
      CHECK_CUDA(cudaEventCreate(&sg.host_viss_ready));
      CHECK_CUDA(cudaEventCreateWithFlags(&sg.pairw_ready, cudaEventDisableTiming));
    }

    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Cacc,(size_t)recon_n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc, 0, (size_t)recon_n_chunk*sizeof(float), ctx[gi].compute_stream));
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk,(size_t)(1LL<<20)*sizeof(float)));

    {
      const int max_plan_blocks = (OP_RECON_PLAN_BL_CHUNK + OP_RECON_TASK_BL - 1) / OP_RECON_TASK_BL;
      const int max_plan_tasks = ctx[gi].ntile_recon * max_plan_blocks;
      cub::CountingInputIterator<int> counting(0);
      for(int pb=0; pb<2; ++pb){
        OperatorPlanBuffer &plan = ctx[gi].recon_plan[pb];
        CHECK_CUDA(cudaMalloc(&plan.d_vv_flags, (size_t)max_plan_tasks * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&plan.d_mixed_flags, (size_t)max_plan_tasks * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&plan.d_vv_tasks, (size_t)max_plan_tasks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&plan.d_mixed_tasks, (size_t)max_plan_tasks * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&plan.d_num_vv, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&plan.d_num_mixed, sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&plan.h_num_vv, sizeof(int)));
        CHECK_CUDA(cudaMallocHost(&plan.h_num_mixed, sizeof(int)));
        size_t temp_bytes_vv = 0, temp_bytes_mixed = 0;
        CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, temp_bytes_vv, counting, plan.d_vv_flags, plan.d_vv_tasks, plan.d_num_vv, max_plan_tasks, ctx[gi].reduce_stream));
        CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr, temp_bytes_mixed, counting, plan.d_mixed_flags, plan.d_mixed_tasks, plan.d_num_mixed, max_plan_tasks, ctx[gi].reduce_stream));
        plan.select_temp_vv_bytes = temp_bytes_vv;
        plan.select_temp_mixed_bytes = temp_bytes_mixed;
        CHECK_CUDA(cudaMalloc(&plan.d_select_temp_vv, temp_bytes_vv));
        CHECK_CUDA(cudaMalloc(&plan.d_select_temp_mixed, temp_bytes_mixed));
        CHECK_CUDA(cudaEventCreate(&plan.ready));
        CHECK_CUDA(cudaEventCreateWithFlags(&plan.compute_done, cudaEventDisableTiming));
        CHECK_CUDA(cudaEventRecord(plan.compute_done, ctx[gi].compute_stream));
      }
    }

    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
  }
  std::cout << "GPU init done in " << t_init.toc_s() << " s\n";

  try_enable_peer_access_between_visible_gpus(ctx);

  BaselineBroadcastInfo binfo;
  query_peer_access(binfo, gpus[gen_gpu_index], ctx);
  ReducerExchangeInfo rinfo;
  query_reducer_exchange(rinfo, gpus[reducer_gpu_index], ctx);
  if(reducer_gpu_index != gen_gpu_index && (rinfo.need_host_stage_collect || rinfo.need_host_stage_scatter)){
    std::cout << "Platform-aware schedule: reducer " << gpus[reducer_gpu_index]
              << " falls back to host staging; pin reducer to generator GPU " << gpus[gen_gpu_index] << "\n";
    reducer_gpu_index = gen_gpu_index;
    query_reducer_exchange(rinfo, gpus[reducer_gpu_index], ctx);
  }
  std::cout << "Generator logical GPU=" << gpus[gen_gpu_index]
            << ", peer-staging-needed=" << (binfo.need_host_stage ? "YES" : "NO") << "\n";
  std::cout << "Reducer logical GPU=" << gpus[reducer_gpu_index]
            << ", collect-host-stage=" << (rinfo.need_host_stage_collect ? "YES" : "NO")
            << ", scatter-host-stage=" << (rinfo.need_host_stage_scatter ? "YES" : "NO") << "\n";
  std::cout << "Recon smallopt: task_bl=" << OP_RECON_TASK_BL
            << ", plan_bl_chunk=" << OP_RECON_PLAN_BL_CHUNK
            << ", tile_pix=" << OP_RECON_TASK_TILE_PIX
            << ", weighted_viss=ON"
            << ", debug_finite=" << OP_RECON_DEBUG_CHECK_FINITE
            << ", skip_zero_weighted_viss=" << OP_RECON_SKIP_ZERO_WEIGHTED_VISS
            << "\n";

  CHECK_CUDA(cudaSetDevice(ctx[reducer_gpu_index].dev));
  for(int slot=0; slot<2; ++slot){
    CHECK_CUDA(cudaMalloc(&ext[reducer_gpu_index].d_reduce_parts[slot], (size_t)G * (size_t)max_segN_half * sizeof(float2)));
  }

  bool need_baseline_host = write_baseline_txt || binfo.need_host_stage;

  OrbitGenCtx gen;
  HostTimer t_gen_persist_init;
  t_gen_persist_init.tic();
  orbit_gen_init(gen, gpus[gen_gpu_index], lambda_m, day_start, max_tlen, orbit_seed);
  std::cout << "Persistent orbit generator init done in " << t_gen_persist_init.toc_s() << " s\n";

  for(int dayid=day_start; dayid<day_start+day_count; ++dayid){
    HostTimer t_day_full;
    t_day_full.tic();
    std::cout << "===== DAY " << dayid << " =====\n";

    for(int gi=0; gi<G; ++gi){
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Cacc, 0, (size_t)ctx[gi].recon_n_chunk*sizeof(float), ctx[gi].compute_stream));
      CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
    }

    std::ofstream ofsViss;
    std::ofstream ofsUVW, ofsXYZa, ofsXYZb;
    if(write_viss){
      std::string fv = out_dir + "Viss" + std::to_string(dayid) + "day" + btag + ".txt";
      ofsViss.open(fv);
      if(!ofsViss.is_open()){
        std::cerr << "ERROR open " << fv << "\n";
        return 1;
      }
    }
    if(write_baseline_txt){
      std::string fuvw = out_dir + "uvw"  + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy1 = out_dir + "xyza" + std::to_string(dayid) + "day" + btag + ".txt";
      std::string fxy2 = out_dir + "xyzb" + std::to_string(dayid) + "day" + btag + ".txt";
      ofsUVW.open(fuvw); ofsXYZa.open(fxy1); ofsXYZb.open(fxy2);
      if(!ofsUVW.is_open() || !ofsXYZa.is_open() || !ofsXYZb.is_open()){
        std::cerr << "ERROR open baseline txt outputs\n";
        return 1;
      }
    }

    HostTimer t_day_segment_loop;
    t_day_segment_loop.tic();
    HostTimer t_stage1_prepare_day;
    t_stage1_prepare_day.tic();
    orbit_gen_prepare_day(gen, dayid);
    double T_prepare_day = t_stage1_prepare_day.toc_s();

    orbit_gen_make_segment_async(gen, 0, seg_t0[0], seg_tlen[0], need_baseline_host);
    broadcast_segment_to_gpus_async(gen, 0, binfo, ctx, need_baseline_host);
    if(segs > 1) orbit_gen_make_segment_async(gen, 1, seg_t0[1], seg_tlen[1], need_baseline_host);

    double stage1_sum_wall_active = 0.0;
    double day_recon_sum = 0.0;

    HostTimer t_rec;
    t_rec.tic();
    for(int s=0; s<segs; ++s){
      int cur = s & 1;
      int nxt = cur ^ 1;

      [[maybe_unused]] HostTimer t_stage1_wall;
      t_stage1_wall.tic();

      orbit_gen_wait_slot(gen, cur);
      const OrbitSegSlot &gcur = gen.slots[cur];

      int seg_t0_cur = gcur.t0;
      int tlen = gcur.tlen;
      int segN = gcur.segN;
      int segN_half = gcur.segN_half;
      if(segN <= 0 || segN_half <= 0) continue;

      if(write_baseline_txt){
        std::vector<float> u_full, v_full, w_full;
        std::vector<float> x1_full, y1_full, z1_full;
        std::vector<float> x2_full, y2_full, z2_full;
        expand_baseline_halfsym_to_full_triplets(gcur.h_u, gcur.h_v, gcur.h_w, segN_half, true,
                                                 u_full, v_full, w_full);
        expand_baseline_halfsym_swap_to_full_triplets(gcur.h_x1, gcur.h_y1, gcur.h_z1,
                                                      gcur.h_x2, gcur.h_y2, gcur.h_z2,
                                                      segN_half,
                                                      x1_full, y1_full, z1_full,
                                                      x2_full, y2_full, z2_full);
        write_txt_3cols_stream(ofsUVW,  u_full.data(),  v_full.data(),  w_full.data(),  (size_t)segN);
        write_txt_3cols_stream(ofsXYZa, x1_full.data(), y1_full.data(), z1_full.data(), (size_t)segN);
        write_txt_3cols_stream(ofsXYZb, x2_full.data(), y2_full.data(), z2_full.data(), (size_t)segN);
      }

      const int BLOCKV=256;
      int gridV=(segN_half + BLOCKV - 1)/BLOCKV;
      size_t shmem=(size_t)viss_tile_pix*4*sizeof(float);

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream=ctx[gi].compute_stream;
        GpuSegSlot &slot = ctx[gi].slots[cur];
        CHECK_CUDA(cudaStreamWaitEvent(stream, slot.ready, 0));
        launch_operator_forward_halfsym_runtime<true>(viss_tile_pix, gridV, BLOCKV, shmem, stream,
          ctx[gi].d_B,ctx[gi].d_l,ctx[gi].d_m,ctx[gi].d_n,ctx[gi].d_nm1,ctx[gi].n_chunk,
          ctx[gi].d_tile_cx_viss, ctx[gi].d_tile_cy_viss, ctx[gi].d_tile_cz_viss,
          ctx[gi].d_tile_cosA_viss, ctx[gi].d_tile_sinA_viss, ctx[gi].ntile_viss,
          slot.d_u, slot.d_v, slot.d_w,
          slot.d_x1, slot.d_y1, slot.d_z1,
          slot.d_x2, slot.d_y2, slot.d_z2,
          segN_half, cosphi,
          slot.d_Vpart);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaEventRecord(slot.viss_stop, stream));
      }

      const int reducer = reducer_gpu_index;
      GpuCtx &rctx = ctx[reducer];
      GpuSegSlot &rslot = rctx.slots[cur];
      CHECK_CUDA(cudaSetDevice(rctx.dev));
      cudaStream_t rstream = rctx.reduce_stream;
      cudaStream_t rxfer = rctx.xfer_stream;
      float2* d_reduce_parts = ext[reducer].d_reduce_parts[cur];
      size_t half_bytes = (size_t)segN_half * sizeof(float2);

      for(int gi=0; gi<G; ++gi){
        float2* dst_part = d_reduce_parts + (size_t)gi * (size_t)max_segN_half;
        if(gi == reducer){
          CHECK_CUDA(cudaStreamWaitEvent(rstream, ctx[gi].slots[cur].viss_stop, 0));
          CHECK_CUDA(cudaMemcpyAsync(dst_part, ctx[gi].slots[cur].d_Vpart, half_bytes,
                                     cudaMemcpyDeviceToDevice, rstream));
        } else if(rinfo.collect_peer_ok[gi]){
          CHECK_CUDA(cudaStreamWaitEvent(rstream, ctx[gi].slots[cur].viss_stop, 0));
          CHECK_CUDA(cudaMemcpyPeerAsync(dst_part, rctx.dev, ctx[gi].slots[cur].d_Vpart, ctx[gi].dev, half_bytes, rstream));
        } else {
          CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
          CHECK_CUDA(cudaStreamWaitEvent(ctx[gi].xfer_stream, ctx[gi].slots[cur].viss_stop, 0));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].slots[cur].h_Vpart, ctx[gi].slots[cur].d_Vpart, half_bytes,
                                     cudaMemcpyDeviceToHost, ctx[gi].xfer_stream));
          CHECK_CUDA(cudaEventRecord(ctx[gi].slots[cur].collect_ready, ctx[gi].xfer_stream));
          CHECK_CUDA(cudaSetDevice(rctx.dev));
          CHECK_CUDA(cudaStreamWaitEvent(rstream, ctx[gi].slots[cur].collect_ready, 0));
          CHECK_CUDA(cudaMemcpyAsync(dst_part, ctx[gi].slots[cur].h_Vpart, half_bytes,
                                     cudaMemcpyHostToDevice, rstream));
        }
      }
      {
        int gb = (segN_half + 255) / 256;
        reduce_phase_halfsym_fused_kernel<<<gb,256,0,rstream>>>(
          d_reduce_parts,
          max_segN_half,
          G,
          rslot.d_w,
          segN_half,
          rslot.d_Viss);
        CHECK_CUDA(cudaPeekAtLastError());
      }
      CHECK_CUDA(cudaEventRecord(rslot.reduce_done, rstream));
      bool need_half_host = write_viss || rinfo.need_host_stage_scatter;
      if(need_half_host){
        CHECK_CUDA(cudaSetDevice(rctx.dev));
        CHECK_CUDA(cudaStreamWaitEvent(rxfer, rslot.reduce_done, 0));
        CHECK_CUDA(cudaMemcpyAsync(rslot.h_Vpart, rslot.d_Viss, half_bytes, cudaMemcpyDeviceToHost, rxfer));
        CHECK_CUDA(cudaEventRecord(rslot.host_viss_ready, rxfer));
      }

      CHECK_CUDA(cudaSetDevice(rctx.dev));
      CHECK_CUDA(cudaStreamWaitEvent(rctx.compute_stream, rslot.reduce_done, 0));
      CHECK_CUDA(cudaEventRecord(rslot.scatter_done, rctx.compute_stream));
      for(int gi=0; gi<G; ++gi){
        if(gi == reducer) continue;
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        cudaStream_t stream = ctx[gi].compute_stream;
        if(rinfo.scatter_peer_ok[gi]){
          CHECK_CUDA(cudaStreamWaitEvent(stream, rslot.reduce_done, 0));
          CHECK_CUDA(cudaMemcpyPeerAsync(ctx[gi].slots[cur].d_Viss, ctx[gi].dev, rslot.d_Viss, rctx.dev, half_bytes, stream));
        } else {
          CHECK_CUDA(cudaStreamWaitEvent(stream, rslot.host_viss_ready, 0));
          CHECK_CUDA(cudaMemcpyAsync(ctx[gi].slots[cur].d_Viss, rslot.h_Vpart, half_bytes,
                                     cudaMemcpyHostToDevice, stream));
        }
        CHECK_CUDA(cudaEventRecord(ctx[gi].slots[cur].scatter_done, stream));
      }

      if(write_viss){
        CHECK_CUDA(cudaSetDevice(rctx.dev));
        CHECK_CUDA(cudaEventSynchronize(rslot.host_viss_ready));
        std::vector<float2> hViss_half((size_t)segN_half);
        std::memcpy(hViss_half.data(), rslot.h_Vpart, half_bytes);
        std::vector<float2> hViss_seg((size_t)segN, make_float2(0,0));
        expand_viss_halfsym_to_full(hViss_half, hViss_seg);
        for(int i=0;i<segN;i++) ofsViss << hViss_seg[i].x << " " << hViss_seg[i].y << "\n";
      }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
        if(gi == reducer){
          CHECK_CUDA(cudaStreamSynchronize(ctx[gi].reduce_stream));
          if(need_half_host) CHECK_CUDA(cudaStreamSynchronize(ctx[gi].xfer_stream));
        }
      }

      double T_wall = t_stage1_wall.toc_s();
      stage1_sum_wall_active += T_wall;

      if(s+1 < segs){
        orbit_gen_wait_slot(gen, nxt);
        broadcast_segment_to_gpus_async(gen, nxt, binfo, ctx, need_baseline_host);
      }
      if(s+2 < segs){
        orbit_gen_make_segment_async(gen, cur, seg_t0[s+2], seg_tlen[s+2], need_baseline_host);
      }

      HostTimer t_rseg;
      t_rseg.tic();
      {
        const int gw = (segN_half + 255) / 256;
        CHECK_CUDA(cudaSetDevice(rctx.dev));
        compute_pair_weight_half_kernel<<<gw,256,0,rctx.reduce_stream>>>(
          rslot.d_u, rslot.d_v, rslot.d_w,
          segN_half,
          ext[reducer].d_dcf, (int)hdcf.size(),
          ext[reducer].d_pairw_half[cur]);
        CHECK_CUDA(cudaPeekAtLastError());
        apply_pair_weight_to_viss_half_kernel<<<gw,256,0,rctx.reduce_stream>>>(
          rslot.d_Viss,
          ext[reducer].d_pairw_half[cur],
          segN_half);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaEventRecord(rslot.pairw_ready, rctx.reduce_stream));

        for(int gi=0; gi<G; ++gi){
          if(gi == reducer) continue;
          CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
          GpuSegSlot &slot = ctx[gi].slots[cur];
          if(rinfo.scatter_peer_ok[gi]){
            CHECK_CUDA(cudaStreamWaitEvent(ctx[gi].xfer_stream, rslot.pairw_ready, 0));
            CHECK_CUDA(cudaMemcpyPeerAsync(ext[gi].d_pairw_half[cur], ctx[gi].dev,
                                           ext[reducer].d_pairw_half[cur], rctx.dev,
                                           (size_t)segN_half * sizeof(float), ctx[gi].xfer_stream));
            CHECK_CUDA(cudaStreamWaitEvent(ctx[gi].xfer_stream, slot.scatter_done, 0));
            apply_pair_weight_to_viss_half_kernel<<<gw,256,0,ctx[gi].xfer_stream>>>(
              slot.d_Viss,
              ext[gi].d_pairw_half[cur],
              segN_half);
            CHECK_CUDA(cudaPeekAtLastError());
            CHECK_CUDA(cudaEventRecord(slot.pairw_ready, ctx[gi].xfer_stream));
          }else{
            compute_pair_weight_half_kernel<<<gw,256,0,ctx[gi].xfer_stream>>>(
              slot.d_u, slot.d_v, slot.d_w,
              segN_half,
              ext[gi].d_dcf, (int)hdcf.size(),
              ext[gi].d_pairw_half[cur]);
            CHECK_CUDA(cudaPeekAtLastError());
            CHECK_CUDA(cudaStreamWaitEvent(ctx[gi].xfer_stream, slot.scatter_done, 0));
            apply_pair_weight_to_viss_half_kernel<<<gw,256,0,ctx[gi].xfer_stream>>>(
              slot.d_Viss,
              ext[gi].d_pairw_half[cur],
              segN_half);
            CHECK_CUDA(cudaPeekAtLastError());
            CHECK_CUDA(cudaEventRecord(slot.pairw_ready, ctx[gi].xfer_stream));
          }
        }
      }

      #pragma omp parallel for num_threads(G)
      for(int gi=0; gi<G; ++gi){
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        (void)run_recon_planaware_segment(ctx[gi], ctx[gi].slots[cur], segN_half, cosphi);
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(ctx[gi].reduce_stream));
      }

      double T_recon_seg = t_rseg.toc_s();
      day_recon_sum += T_recon_seg;
      std::cout << "[day " << dayid << " seg " << (s+1) << "/" << segs
                << "] t0=" << seg_t0_cur
                << " segN=" << segN
                << " segN_half=" << segN_half
                << " VissTime=" << T_wall << "s"
                << " ReconSegTime=" << T_recon_seg << "s"
                << " mode=direct3d_halfsym_tilecone_balance\n";
    }

    double T_day_segment_loop = t_day_segment_loop.toc_s();

    double T_recon_day_wall = t_rec.toc_s();
    std::cout<<"Direct 3D recon done in "<<T_recon_day_wall<<" s\n";

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
      for(const auto &rb : recon_blocks){
        int gi = rb.owner_gpu;
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        long long off = rb.owner_off;
        long long remain = rb.len;
        while(remain > 0){
          int cur=(int)std::min((long long)CH, remain);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          for(int i=0;i<cur;i++) ofs<<ctx[gi].h_chunk[i]<<"\n";
          off += cur;
          remain -= cur;
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
      for(const auto &rb : recon_blocks){
        int gi = rb.owner_gpu;
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        long long off = rb.owner_off;
        long long remain = rb.len;
        while(remain > 0){
          int cur=(int)std::min((long long)CH, remain);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Cacc+off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          ofs.write(reinterpret_cast<const char*>(ctx[gi].h_chunk), (std::streamsize)cur*sizeof(float));
          off += cur;
          remain -= cur;
        }
      }
      ofs.close();
    }
    double T_day_output = t_w.toc_s();
    std::cout<<"Wrote "<<outC<<" in "<<T_day_output<<" s\n";
    std::cout<<"[day "<<dayid<<" Full summary] T_prepare_day="<<T_prepare_day
             <<"s T_stage1_active_sum="<<stage1_sum_wall_active
             <<"s T_recon_sum="<<day_recon_sum
             <<"s T_recon_day_wall="<<T_recon_day_wall
             <<"s T_day_segment_loop="<<T_day_segment_loop
             <<"s T_output_C="<<T_day_output
             <<"s T_full_day_wall="<<t_day_full.toc_s()<<"s\n";

    if(write_viss) ofsViss.close();
    if(write_baseline_txt){ ofsUVW.close(); ofsXYZa.close(); ofsXYZb.close(); }
  }

  orbit_gen_destroy(gen);

  #pragma omp parallel for num_threads(G)
  for(int gi=0; gi<G; ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    if(ctx[gi].h_chunk) cudaFreeHost(ctx[gi].h_chunk);
    for(int pb=0; pb<2; ++pb){
      OperatorPlanBuffer &plan = ctx[gi].recon_plan[pb];
      if(plan.compute_done) cudaEventDestroy(plan.compute_done);
      if(plan.ready) cudaEventDestroy(plan.ready);
      if(plan.d_select_temp_mixed) cudaFree(plan.d_select_temp_mixed);
      if(plan.d_select_temp_vv) cudaFree(plan.d_select_temp_vv);
      if(plan.h_num_mixed) cudaFreeHost(plan.h_num_mixed);
      if(plan.h_num_vv) cudaFreeHost(plan.h_num_vv);
      if(plan.d_num_mixed) cudaFree(plan.d_num_mixed);
      if(plan.d_num_vv) cudaFree(plan.d_num_vv);
      if(plan.d_mixed_tasks) cudaFree(plan.d_mixed_tasks);
      if(plan.d_vv_tasks) cudaFree(plan.d_vv_tasks);
      if(plan.d_mixed_flags) cudaFree(plan.d_mixed_flags);
      if(plan.d_vv_flags) cudaFree(plan.d_vv_flags);
    }

    if(ext[gi].d_reduce_parts[1]) cudaFree(ext[gi].d_reduce_parts[1]);
    if(ext[gi].d_reduce_parts[0]) cudaFree(ext[gi].d_reduce_parts[0]);
    if(ext[gi].d_pairw_half[1]) cudaFree(ext[gi].d_pairw_half[1]);
    if(ext[gi].d_pairw_half[0]) cudaFree(ext[gi].d_pairw_half[0]);
    if(ext[gi].d_dcf) cudaFree(ext[gi].d_dcf);

    if(ctx[gi].d_Cacc) cudaFree(ctx[gi].d_Cacc);

    for(int slot=0; slot<2; ++slot){
      GpuSegSlot &sg = ctx[gi].slots[slot];
      if(sg.pairw_ready) cudaEventDestroy(sg.pairw_ready);
      if(sg.host_viss_ready) cudaEventDestroy(sg.host_viss_ready);
      if(sg.scatter_done) cudaEventDestroy(sg.scatter_done);
      if(sg.reduce_done) cudaEventDestroy(sg.reduce_done);
      if(sg.collect_ready) cudaEventDestroy(sg.collect_ready);
      if(sg.viss_stop) cudaEventDestroy(sg.viss_stop);
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
    }

    if(ctx[gi].d_tile_sinA_viss) cudaFree(ctx[gi].d_tile_sinA_viss);
    if(ctx[gi].d_tile_cosA_viss) cudaFree(ctx[gi].d_tile_cosA_viss);
    if(ctx[gi].d_tile_cz_viss) cudaFree(ctx[gi].d_tile_cz_viss);
    if(ctx[gi].d_tile_cy_viss) cudaFree(ctx[gi].d_tile_cy_viss);
    if(ctx[gi].d_tile_cx_viss) cudaFree(ctx[gi].d_tile_cx_viss);
    if(ctx[gi].d_tile_sinA_recon) cudaFree(ctx[gi].d_tile_sinA_recon);
    if(ctx[gi].d_tile_cosA_recon) cudaFree(ctx[gi].d_tile_cosA_recon);
    if(ctx[gi].d_tile_cz_recon) cudaFree(ctx[gi].d_tile_cz_recon);
    if(ctx[gi].d_tile_cy_recon) cudaFree(ctx[gi].d_tile_cy_recon);
    if(ctx[gi].d_tile_cx_recon) cudaFree(ctx[gi].d_tile_cx_recon);
    if(ctx[gi].d_n) cudaFree(ctx[gi].d_n);
    if(ctx[gi].d_nm1) cudaFree(ctx[gi].d_nm1);
    if(ctx[gi].d_m) cudaFree(ctx[gi].d_m);
    if(ctx[gi].d_l) cudaFree(ctx[gi].d_l);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].reduce_stream) cudaStreamDestroy(ctx[gi].reduce_stream);
    if(ctx[gi].xfer_stream) cudaStreamDestroy(ctx[gi].xfer_stream);
    if(ctx[gi].compute_stream) cudaStreamDestroy(ctx[gi].compute_stream);
  }

  return 0;
}
