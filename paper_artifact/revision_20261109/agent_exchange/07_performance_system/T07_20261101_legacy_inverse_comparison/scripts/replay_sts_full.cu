#include "../../../01_common_operator/T01_20261101_production_replay/source/common_mirror_tile_vissfast.hpp"
#include "../../../01_common_operator/T01_20261101_production_replay/source/orbit_online_mirror_tile_vissfast.cuh"
#include "../../../01_common_operator/T01_20261101_production_replay/source/viss_recon_kernel_mirror_tile_vissfast.cuh"
#include "../../../01_common_operator/T01_20261101_production_replay/scripts/replay_common.hpp"
#include <openssl/sha.h>
#include <iomanip>
#include <memory>
#include <sstream>

struct PreparedSegment {
  int seg, t0, tlen, nh;
  std::array<std::vector<float>,11> full;
  std::vector<float2> vis;
  std::vector<float> half_u,half_v;
};

static std::string hex_sha(const unsigned char* d) {
  std::ostringstream s; s<<std::hex<<std::setfill('0');
  for(int i=0;i<SHA256_DIGEST_LENGTH;i++)s<<std::setw(2)<<(unsigned)d[i];
  return s.str();
}
template<class T> static std::string scan_array(const T* device_ptr,long long n) {
  SHA256_CTX context;SHA256_Init(&context);long long count=0,nonfinite=0,zeros=0;
  long double sum=0,sum2=0;double lo=INFINITY,hi=-INFINITY;
  std::vector<T> part(1<<20);
  for(long long base=0;base<n;base+=(1<<20)) {
    size_t take=(size_t)std::min<long long>(1<<20,n-base);
    CHECK_CUDA(cudaMemcpy(part.data(),device_ptr+base,take*sizeof(T),cudaMemcpyDeviceToHost));
    SHA256_Update(&context,part.data(),take*sizeof(T));
    for(size_t i=0;i<take;i++) {
      double value=(double)part[i];count++;if(!std::isfinite(value)){nonfinite++;continue;}
      zeros+=(value==0);lo=std::min(lo,value);hi=std::max(hi,value);sum+=value;sum2+=(long double)value*value;
    }
  }
  unsigned char digest[SHA256_DIGEST_LENGTH];SHA256_Final(digest,&context);
  std::ostringstream s;s<<std::setprecision(17)<<"{\"count\":"<<count<<",\"nonfinite\":"<<nonfinite
    <<",\"zeros\":"<<zeros<<",\"min\":"<<lo<<",\"max\":"<<hi<<",\"sum\":"<<(double)sum
    <<",\"sum_squares\":"<<(double)sum2<<",\"sha256\":\""<<hex_sha(digest)<<"\"}";
  if(nonfinite)throw std::runtime_error("nonfinite full output");return s.str();
}
static double elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));return ms*.001;}

int main(int argc,char** argv){try{
  const std::string gpu=get_arg(argc,argv,"--gpus","3"),cache=get_arg(argc,argv,"--cache_root",""),out=get_arg(argc,argv,"--out_dir","");
  const std::string phase=get_arg(argc,argv,"--rep_phase","grid");
  const int nside=to_int(get_arg(argc,argv,"--nside","4096"),4096),day=to_int(get_arg(argc,argv,"--day","1"),1),nseg=to_int(get_arg(argc,argv,"--max_segments","10"),10);
  if(cache.empty()||out.empty()||nside!=4096||day!=1||nseg<1||nseg>10||(phase!="grid"&&phase!="actual"))throw std::runtime_error("invalid frozen benchmark arguments");
  setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID",1);setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);CHECK_CUDA(cudaSetDevice(0));
  ensure_dir(out);disk_guard(out,100ULL*1024*1024*1024);
  HostTimer load_timer;load_timer.tic();std::vector<PreparedSegment> segments;int expected_t0=0,maxnh=0;
  for(int sid=1;sid<=nseg;sid++){
    std::ifstream f(cache+"/day1_seg"+std::to_string(sid)+".bin",std::ios::binary);ReplayHeader h{};f.read((char*)&h,sizeof(h));
    if(!f.good()||std::memcmp(h.magic,"PREPLAY1",8)||h.nside!=4096||h.day!=1||h.seg!=sid||h.t0!=expected_t0||h.nhalf!=28*h.tlen||h.forward_gpus!=1)throw std::runtime_error("cache identity");
    std::vector<ReplayRow> rows(h.nhalf);f.read((char*)rows.data(),rows.size()*sizeof(ReplayRow));if(!f.good())throw std::runtime_error("cache short read");
    PreparedSegment a;a.seg=sid;a.t0=h.t0;a.tlen=h.tlen;a.nh=h.nhalf;
    for(auto& v:a.full)v.assign(2*h.nhalf,0);a.vis.resize(h.nhalf);a.half_u.resize(h.nhalf);a.half_v.resize(h.nhalf);
    for(int j=0;j<h.nhalf;j++){
      const auto r=rows[j];int pos=half_to_full_pos_idx(j),neg=half_to_full_neg_idx(j);
      const float xyz[11]={r.u,r.v,r.w,r.x1,r.y1,r.z1,r.inv1,r.x2,r.y2,r.z2,r.inv2};
      for(int k=0;k<11;k++)a.full[k][pos]=xyz[k];a.full[0][neg]=-r.u;a.full[1][neg]=-r.v;a.full[2][neg]=-r.w;
      for(int k=0;k<4;k++){a.full[3+k][neg]=xyz[7+k];a.full[7+k][neg]=xyz[3+k];}
      a.vis[j]=make_float2(r.re,r.im);a.half_u[j]=r.u;a.half_v[j]=r.v;
    }
    expected_t0+=h.tlen;maxnh=std::max(maxnh,h.nhalf);segments.push_back(std::move(a));
  }
  const double cache_s=load_timer.toc_s();const long long np=12LL*nside*nside;const int block=256,grid=(int)((np+255)/256);
  HostTimer init_timer;init_timer.tic();size_t free_before=0,total_bytes=0;
  CHECK_CUDA(cudaMemGetInfo(&free_before,&total_bytes));
  const uint64_t estimated_bytes=(uint64_t)np*28+512ULL*1024*1024;
  if(free_before<estimated_bytes+2ULL*1024*1024*1024)throw std::runtime_error("allocation preflight: need estimated STS arrays plus 2GiB margin");
  {std::ofstream f(out+"/allocation_preflight.json");f<<"{\"free_before_bytes\":"<<free_before<<",\"total_bytes\":"<<total_bytes<<",\"estimated_bytes\":"<<estimated_bytes<<",\"reserve_bytes\":2147483648,\"is_peak_measurement\":false}\n";}
  float *l,*m,*n,*N,*S;uint32_t *W,*T;
  CHECK_CUDA(cudaMalloc(&l,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&m,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&n,np*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&N,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&S,np*sizeof(float)));CHECK_CUDA(cudaMalloc(&W,np*sizeof(uint32_t)));CHECK_CUDA(cudaMalloc(&T,np*sizeof(uint32_t)));
  pix2lmn_nest_kernel<<<grid,block>>>(nside,0,(int)np,l,m,n);
  CHECK_CUDA(cudaMemset(N,0,np*sizeof(float)));CHECK_CUDA(cudaMemset(W,0,np*sizeof(uint32_t)));
  std::array<float*,11> d;for(auto& v:d)CHECK_CUDA(cudaMalloc(&v,2*maxnh*sizeof(float)));
  float2 *dv,*repV;float4 *p1,*p2;int *keys,*reps;float *u,*v;
  CHECK_CUDA(cudaMalloc(&dv,maxnh*sizeof(float2)));CHECK_CUDA(cudaMalloc(&repV,maxnh*sizeof(float2)));
  CHECK_CUDA(cudaMalloc(&p1,maxnh*sizeof(float4)));CHECK_CUDA(cudaMalloc(&p2,maxnh*sizeof(float4)));
  CHECK_CUDA(cudaMalloc(&keys,maxnh*sizeof(int)));CHECK_CUDA(cudaMalloc(&reps,maxnh*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&u,maxnh*sizeof(float)));CHECK_CUDA(cudaMalloc(&v,maxnh*sizeof(float)));
  cudaEvent_t ha,hb,ka,kb;for(auto p:{&ha,&hb,&ka,&kb})CHECK_CUDA(cudaEventCreate(p));
  CHECK_CUDA(cudaDeviceSynchronize());const double init_s=init_timer.toc_s();size_t free_after_init=0;CHECK_CUDA(cudaMemGetInfo(&free_after_init,&total_bytes));
  const float theta=asinf(1737.1e3f/(1737.1e3f+300e3f)),cosphi=cosf((float)M_PI-theta);
  std::ofstream csv(out+"/segments.csv");csv<<"segment,half,groups,fa,fb,du,RES,input_h2d_cuda_s,weight_or_group_wall_s,recon_wall_s,recon_cuda_s,segment_wall_s,boundary_occupied_bytes\n"<<std::setprecision(17);
  HostTimer loop_timer;loop_timer.tic();double sum_kernel=0,sum_h2d=0;
  for(const auto& a:segments){
    HostTimer seg_timer;seg_timer.tic();CHECK_CUDA(cudaEventRecord(ha));
    for(int j=0;j<11;j++)CHECK_CUDA(cudaMemcpy(d[j],a.full[j].data(),2*a.nh*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dv,a.vis.data(),a.nh*sizeof(float2),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaEventRecord(hb));CHECK_CUDA(cudaEventSynchronize(hb));const double copy_s=elapsed(ha,hb);sum_h2d+=copy_s;
    HostTimer group_timer;group_timer.tic();float fa=0,fb=0;double denom=0;
    compute_fa_fb_host_matlab_no_intercept(a.full[0].data(),a.full[1].data(),a.full[2].data(),a.nh*2,fa,fb,denom);
    const float lmmax=std::max(std::sqrt(1.0f+fa*fa),std::sqrt(1.0f+fb*fb));
    const float temp=std::ceil(2.0f*100e3f/30.0f*2.0f*lmmax);
    const int RES=(int)(temp+2+1-std::fmod(temp,2.0f)),half=(RES-1)/2;const float du=2.0f*100e3f/30.0f/(RES-1);
    GroupArtifactsHost grp;build_groups_device0_half(0,0,d[0],d[1],dv,a.nh,du,RES,half,1,2,grp);
    if(grp.nuniq<=0||grp.rep_half_idx.size()!=(size_t)grp.nuniq)throw std::runtime_error("empty/missing representatives");
    std::vector<float> hu(grp.nuniq),hv(grp.nuniq);
    for(int q=0;q<grp.nuniq;q++){
      int key=grp.keys_unique[q];if(key<=0)throw std::runtime_error("out of grid");int tmp=key-1,U=tmp/RES,V=tmp-U*RES;
      hu[q]=phase=="actual"?a.half_u[grp.rep_half_idx[q]]:(U-half)*du;
      hv[q]=phase=="actual"?a.half_v[grp.rep_half_idx[q]]:(V-half)*du;
    }
    CHECK_CUDA(cudaMemcpy(keys,grp.keys_unique.data(),grp.nuniq*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(reps,grp.rep_half_idx.data(),grp.nuniq*sizeof(int),cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(u,hu.data(),grp.nuniq*sizeof(float),cudaMemcpyHostToDevice));CHECK_CUDA(cudaMemcpy(v,hv.data(),grp.nuniq*sizeof(float),cudaMemcpyHostToDevice));
    gather_rep_half_inputs<<<(grp.nuniq+255)/256,256>>>(reps,grp.nuniq,dv,d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],repV,p1,p2);
    CHECK_CUDA(cudaDeviceSynchronize());const double group_s=group_timer.toc_s();
    HostTimer recon_timer;recon_timer.tic();CHECK_CUDA(cudaMemset(S,0,np*sizeof(float)));CHECK_CUDA(cudaMemset(T,0,np*sizeof(uint32_t)));
    CHECK_CUDA(cudaEventRecord(ka));recon_seg_blockage_half_rep<64><<<grid,block>>>(np,l,m,n,keys,u,v,repV,p1,p2,grp.nuniq,fa,fb,cosphi,S,T);CHECK_CUDA(cudaEventRecord(kb));
    add_inplace<<<grid,block>>>(N,S,np);add_inplace_u32<<<grid,block>>>(W,T,np);CHECK_CUDA(cudaPeekAtLastError());CHECK_CUDA(cudaDeviceSynchronize());
    const double recon_s=recon_timer.toc_s(),kernel_s=elapsed(ka,kb);sum_kernel+=kernel_s;
    size_t boundary_free=0;CHECK_CUDA(cudaMemGetInfo(&boundary_free,&total_bytes));
    csv<<a.seg<<","<<a.nh<<","<<grp.nuniq<<","<<fa<<","<<fb<<","<<du<<","<<RES<<","<<copy_s<<","<<group_s<<","<<recon_s<<","<<kernel_s<<","<<seg_timer.toc_s()<<","<<total_bytes-boundary_free<<"\n";csv.flush();
    std::cout<<"FULL_STS_SEG_DONE "<<a.seg<<"/"<<nseg<<"\n";std::cout.flush();
  }
  CHECK_CUDA(cudaDeviceSynchronize());const double loop_s=loop_timer.toc_s();
  HostTimer scan_timer;scan_timer.tic();auto ids=sample_ids(nside,8192);auto did=device_copy(ids);float* sampleN;uint32_t* sampleW;
  CHECK_CUDA(cudaMalloc(&sampleN,8192*sizeof(float)));CHECK_CUDA(cudaMalloc(&sampleW,8192*sizeof(uint32_t)));
  gather_samples<<<32,256>>>(did,8192,0,np,N,sampleN);gather_samples_u32<<<32,256>>>(did,8192,0,np,W,sampleW);
  write_values(out+"/sample_ids.u32",ids);write_values(out+"/sample_N.f32",host_copy(sampleN,8192));write_values(out+"/sample_W.u32",host_copy(sampleW,8192));
  const std::string raw_stats=scan_array(N,np),weight_stats=scan_array(W,np);const double scan_raw_s=scan_timer.toc_s();
  HostTimer final_timer;final_timer.tic();normalize_by_weight<<<grid,block>>>(N,W,np);CHECK_CUDA(cudaPeekAtLastError());CHECK_CUDA(cudaDeviceSynchronize());const double finalize_s=final_timer.toc_s();
  scan_timer.tic();const std::string map_stats=scan_array(N,np);const double scan_map_s=scan_timer.toc_s();
  std::ofstream stats(out+"/full_stats.json");stats<<"{\"rawN\":"<<raw_stats<<",\"W\":"<<weight_stats<<",\"C\":"<<map_stats<<"}\n";
  std::ofstream meta(out+"/metrics.json");meta<<std::setprecision(17)<<"{\"method\":\"sts_"<<phase<<"\",\"day\":1,\"nside\":4096,\"pixels\":"<<np<<",\"segments\":"<<nseg
    <<",\"cache_load_decode_wall_s\":"<<cache_s<<",\"init_wall_s\":"<<init_s<<",\"inverse_loop_wall_s\":"<<loop_s
    <<",\"output_finalize_wall_s\":"<<finalize_s<<",\"loop_plus_finalize_wall_s\":"<<loop_s+finalize_s
    <<",\"recon_cuda_sum_s\":"<<sum_kernel<<",\"input_h2d_cuda_sum_s\":"<<sum_h2d<<",\"scan_wall_s\":"<<scan_raw_s+scan_map_s
    <<",\"init_boundary_occupied_bytes\":"<<total_bytes-free_after_init<<",\"memory_scope\":\"phase boundaries, not instantaneous peak\""
    <<",\"output_map_written\":false,\"timer_scope\":\"cached inverse only: geometry/V H2D, fit/group/metadata, rep kernel and accumulation; finalize separate; cache/decode, init and full SHA scans excluded\"}\n";
  std::cout<<"FULL_STS_DONE loop="<<loop_s<<" finalize="<<finalize_s<<"\n";return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<"\n";return 2;}}
