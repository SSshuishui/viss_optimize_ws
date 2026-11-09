#if LEGACY_FASTPATH
#include "../source/legacy_combined/common.hpp"
#include "../source/legacy_combined/orbit_online.cuh"
#include "../source/legacy_combined/viss_recon_kernel.cuh"
#else
#include "../source/controlled_fixed/common.hpp"
#include "../source/controlled_fixed/orbit_online.cuh"
#include "../source/controlled_fixed/viss_recon_kernel.cuh"
#endif
#include <openssl/sha.h>
#include <array>
#include <stdexcept>
#include <chrono>
#include <climits>

struct CacheHeader{char magic[8];int32_t nside,day,seg,t0,tlen,nhalf,G;};
struct Row{float u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im;};
static_assert(sizeof(CacheHeader)==36 && sizeof(Row)==60,"frozen cache layout");
struct HostSegment{CacheHeader hdr;std::array<std::vector<float>,9>a;std::vector<float2>vis;};
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
template<class T>static T* alloc(size_t n){T*d=nullptr;CHECK_CUDA(cudaMalloc(&d,n*sizeof(T)));return d;}
template<class T>static void write_vec(const std::string&p,const std::vector<T>&v){std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));if(!f.good())throw std::runtime_error("output "+p);}
template<class T>static std::vector<T> read_vec(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto bytes=f.tellg();if(bytes<0 || (size_t)bytes%sizeof(T))throw std::runtime_error("shape "+p);std::vector<T>v((size_t)bytes/sizeof(T));f.seekg(0);f.read((char*)v.data(),bytes);if(!f.good())throw std::runtime_error("read "+p);return v;}
static std::string hex_sha(const unsigned char*d){std::ostringstream o;for(int i=0;i<SHA256_DIGEST_LENGTH;i++)o<<std::hex<<std::setw(2)<<std::setfill('0')<<(unsigned int)d[i];return o.str();}
static double event_s(cudaEvent_t a,cudaEvent_t b){float ms=0;CHECK_CUDA(cudaEventElapsedTime(&ms,a,b));return ms*.001;}
__global__ void sample_gather(const float*a,const uint32_t*ids,int n,float*out){int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<n)out[p]=a[ids[p]];}

int main(int argc,char**argv){try{
 std::string gpu=get_arg(argc,argv,"--gpus",""),cache=get_arg(argc,argv,"--cache_root",""),out=get_arg(argc,argv,"--out_dir","");
 int nside=std::stoi(get_arg(argc,argv,"--nside","4096")),day=std::stoi(get_arg(argc,argv,"--day","1")),nseg=std::stoi(get_arg(argc,argv,"--max_segments","10"));
 std::string sampledir=get_arg(argc,argv,"--sample_dir",cache+"/../runs/day1_s10_g1_n8192");
 std::string dcfpath=get_arg(argc,argv,"--dcf_bin","/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin");
 if(gpu.empty()||gpu.find(',')!=std::string::npos||cache.empty()||out.empty()||nside!=4096||day!=1||nseg!=10)throw std::runtime_error("explicit single GPU, cache/out paths and nside4096/day1/10 original segments required");
 setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID",1);setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);CHECK_CUDA(cudaSetDevice(0));
 const long long np=12LL*nside*nside;const int nt=(int)((np+255)/256);const int nbmax=195468;
 auto tcache=Clock::now();std::vector<HostSegment>host_segments;host_segments.reserve(nseg);
 for(int sid=1;sid<=nseg;sid++){
  std::ifstream f(cache+"/day1_seg"+std::to_string(sid)+".bin",std::ios::binary);HostSegment h;f.read((char*)&h.hdr,sizeof(h.hdr));
  if(!f.good()||std::memcmp(h.hdr.magic,"PREPLAY1",8)||h.hdr.nside!=nside||h.hdr.day!=day||h.hdr.seg!=sid||h.hdr.nhalf!=nbmax||h.hdr.G!=1||h.hdr.tlen!=6981||h.hdr.t0!=(sid-1)*6981)throw std::runtime_error("cache identity "+std::to_string(sid));
  std::vector<Row>rows(nbmax);f.read((char*)rows.data(),rows.size()*sizeof(Row));if(!f.good())throw std::runtime_error("cache body");
  for(auto&v:h.a)v.resize(nbmax);h.vis.resize(nbmax);
  for(int j=0;j<nbmax;j++){auto r=rows[j];float v[9]={r.u,r.v,r.w,r.x1,r.y1,r.z1,r.x2,r.y2,r.z2};for(int k=0;k<9;k++)h.a[k][j]=v[k];h.vis[j]=make_float2(r.re,r.im);}
  host_segments.push_back(std::move(h));
 }
 DcfMbBinPayload payload;if(!load_dcf_mb_bin(dcfpath,payload)||payload.btag!="10M"||payload.dcf_days!=450||payload.orbit_seed!=42)throw std::runtime_error("DCF identity");
 auto ids=read_vec<uint32_t>(sampledir+"/ids.u32");if(ids.size()!=8192)throw std::runtime_error("8192 sample IDs required");for(auto id:ids)if(id>=(uint64_t)np)throw std::runtime_error("sample ID range");
 double cache_s=elapsed(tcache);
 auto tinit=Clock::now();size_t free0=0,totalmem=0;CHECK_CUDA(cudaMemGetInfo(&free0,&totalmem));
 const size_t estimated_min=(size_t)np*16+(size_t)nt*20+(size_t)nbmax*48+256*1024*1024ULL;
 if(free0<estimated_min+512*1024*1024ULL)throw std::runtime_error("full-output minimum allocation guard, no smaller-output fallback");
 auto dl=alloc<float>(np),dm=alloc<float>(np),dn=alloc<float>(np),C=alloc<float>(np);
 std::array<float*,5>tc;for(auto&p:tc)p=alloc<float>(nt);
 cudaStream_t compute,xfer,reduce;CHECK_CUDA(cudaStreamCreate(&compute));CHECK_CUDA(cudaStreamCreate(&xfer));CHECK_CUDA(cudaStreamCreate(&reduce));
 pix2lmn_nest_kernel<<<((int)np+255)/256,256,0,compute>>>(nside,0,(int)np,dl,dm,dn);
 launch_build_tile_cone_meta_runtime(256,compute,dl,dm,dn,np,tc[0],tc[1],tc[2],tc[3],tc[4],nt);
 CHECK_CUDA(cudaMemsetAsync(C,0,np*sizeof(float),compute));
 std::array<float*,9>d;for(auto&p:d)p=alloc<float>(nbmax);auto dv=alloc<float2>(nbmax);auto q=alloc<float>(nbmax),dcf=alloc<float>(payload.dcf.size());
 CHECK_CUDA(cudaMemcpy(dcf,payload.dcf.data(),payload.dcf.size()*sizeof(float),cudaMemcpyHostToDevice));
 cudaEvent_t hs,he,ps,pe,rs,re;for(auto p:{&hs,&he,&ps,&pe,&rs,&re})CHECK_CUDA(cudaEventCreate(p));
 size_t plan_bytes=0;
#if LEGACY_FASTPATH
 GpuCtx ctx;ctx.dev=0;ctx.compute_stream=compute;ctx.reduce_stream=reduce;ctx.recon_n_chunk=np;ctx.ntile_recon=nt;ctx.d_l_recon=dl;ctx.d_m_recon=dm;ctx.d_n_recon=dn;ctx.d_Cacc=C;
 ctx.d_tile_cx_recon=tc[0];ctx.d_tile_cy_recon=tc[1];ctx.d_tile_cz_recon=tc[2];ctx.d_tile_cosA_recon=tc[3];ctx.d_tile_sinA_recon=tc[4];
 GpuSegSlot slot;slot.d_u=d[0];slot.d_v=d[1];slot.d_w=d[2];slot.d_x1=d[3];slot.d_y1=d[4];slot.d_z1=d[5];slot.d_x2=d[6];slot.d_y2=d[7];slot.d_z2=d[8];slot.d_Viss=dv;
 CHECK_CUDA(cudaEventCreateWithFlags(&slot.pairw_ready,cudaEventDisableTiming));
 const int maxtasks=nt*((OP_RECON_PLAN_BL_CHUNK+OP_RECON_TASK_BL-1)/OP_RECON_TASK_BL);cub::CountingInputIterator<int>counting(0);
 for(int pb=0;pb<2;pb++){
  auto& p=ctx.recon_plan[pb];p.d_vv_flags=alloc<unsigned char>(maxtasks);p.d_mixed_flags=alloc<unsigned char>(maxtasks);p.d_vv_tasks=alloc<int>(maxtasks);p.d_mixed_tasks=alloc<int>(maxtasks);p.d_num_vv=alloc<int>(1);p.d_num_mixed=alloc<int>(1);CHECK_CUDA(cudaMallocHost(&p.h_num_vv,sizeof(int)));CHECK_CUDA(cudaMallocHost(&p.h_num_mixed,sizeof(int)));
  CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr,p.select_temp_vv_bytes,counting,p.d_vv_flags,p.d_vv_tasks,p.d_num_vv,maxtasks,reduce));CHECK_CUDA(cub::DeviceSelect::Flagged(nullptr,p.select_temp_mixed_bytes,counting,p.d_mixed_flags,p.d_mixed_tasks,p.d_num_mixed,maxtasks,reduce));
  CHECK_CUDA(cudaMalloc(&p.d_select_temp_vv,p.select_temp_vv_bytes));CHECK_CUDA(cudaMalloc(&p.d_select_temp_mixed,p.select_temp_mixed_bytes));CHECK_CUDA(cudaEventCreate(&p.ready));CHECK_CUDA(cudaEventCreateWithFlags(&p.compute_done,cudaEventDisableTiming));CHECK_CUDA(cudaEventRecord(p.compute_done,compute));
  plan_bytes+=(size_t)maxtasks*10+8+p.select_temp_vv_bytes+p.select_temp_mixed_bytes;
 }
#endif
 CHECK_CUDA(cudaDeviceSynchronize());size_t free_init=0;CHECK_CUDA(cudaMemGetInfo(&free_init,&totalmem));size_t minfree=free_init;double init_s=elapsed(tinit);
 std::ofstream initreport(out+"/allocation_preflight.json");initreport<<"{\"nside\":4096,\"full_pixels\":"<<np<<",\"estimated_min_bytes\":"<<estimated_min<<",\"device_occupied_bytes\":"<<(totalmem-free_init)<<",\"plan_bytes\":"<<plan_bytes<<",\"allocation_pass\":true}\n";initreport.close();
 float R=1737.1e3f,H=300e3f,cp=cosf((float)M_PI-asinf(R/(R+H)));
 struct SegmentTiming{int sid;double h2d,prep,recon,span,wall;long long vv,mixed;};std::vector<SegmentTiming>timings;
 CHECK_CUDA(cudaDeviceSynchronize());auto tloop=Clock::now();
 for(int sid=1;sid<=nseg;sid++){
  auto tseg=Clock::now();auto&h=host_segments[sid-1];
  CHECK_CUDA(cudaEventRecord(hs,xfer));for(int k=0;k<9;k++)CHECK_CUDA(cudaMemcpyAsync(d[k],h.a[k].data(),nbmax*sizeof(float),cudaMemcpyHostToDevice,xfer));CHECK_CUDA(cudaMemcpyAsync(dv,h.vis.data(),nbmax*sizeof(float2),cudaMemcpyHostToDevice,xfer));CHECK_CUDA(cudaEventRecord(he,xfer));CHECK_CUDA(cudaEventSynchronize(he));
  auto tprep=Clock::now();CHECK_CUDA(cudaEventRecord(ps,compute));normalize3_inplace_kernel<<<(nbmax+255)/256,256,0,compute>>>(d[3],d[4],d[5],nbmax);normalize3_inplace_kernel<<<(nbmax+255)/256,256,0,compute>>>(d[6],d[7],d[8],nbmax);compute_pair_weight_half_kernel<<<(nbmax+255)/256,256,0,compute>>>(d[0],d[1],d[2],nbmax,dcf,payload.dcf.size(),q);
#if LEGACY_FASTPATH
  apply_pair_weight_to_viss_half_kernel<<<(nbmax+255)/256,256,0,compute>>>(dv,q,nbmax);CHECK_CUDA(cudaEventRecord(slot.pairw_ready,compute));
#endif
  CHECK_CUDA(cudaEventRecord(pe,compute));CHECK_CUDA(cudaEventSynchronize(pe));double prep_wall=elapsed(tprep);
  auto trecon=Clock::now();CHECK_CUDA(cudaEventRecord(rs,compute));long long vv=0,mixed=0;
#if LEGACY_FASTPATH
  auto stats=run_recon_planaware_segment(ctx,slot,nbmax,cp);vv=stats.vv_tasks;mixed=stats.mixed_tasks;
#else
  recon_3d_direct_halfsym_tilecone_real<256,128,true><<<nt,256,0,compute>>>(np,dl,dm,dn,tc[0],tc[1],tc[2],tc[3],tc[4],nt,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],q,dv,nbmax,cp,C);
#endif
  CHECK_CUDA(cudaEventRecord(re,compute));CHECK_CUDA(cudaDeviceSynchronize());double recon_wall=elapsed(trecon);size_t freenow=0;CHECK_CUDA(cudaMemGetInfo(&freenow,&totalmem));minfree=std::min(minfree,freenow);
  timings.push_back({sid,event_s(hs,he),prep_wall,recon_wall,event_s(rs,re),elapsed(tseg),vv,mixed});std::cout<<"INVERSE_SEG_DONE "<<sid<<"/10\n";std::cout.flush();
 }
 CHECK_CUDA(cudaDeviceSynchronize());double loop_s=elapsed(tloop);
 auto tsample=Clock::now();auto did=alloc<uint32_t>(ids.size());auto ds=alloc<float>(ids.size());CHECK_CUDA(cudaMemcpy(did,ids.data(),ids.size()*sizeof(uint32_t),cudaMemcpyHostToDevice));sample_gather<<<32,256>>>(C,did,8192,ds);std::vector<float>sample(8192);CHECK_CUDA(cudaMemcpy(sample.data(),ds,sample.size()*sizeof(float),cudaMemcpyDeviceToHost));write_vec(out+"/sample_ids.u32",ids);write_vec(out+"/sample_N.f32",sample);double sample_s=elapsed(tsample);
 auto tscan=Clock::now();const size_t CH=1<<20;std::vector<float>buf(CH);SHA256_CTX sh;SHA256_Init(&sh);long long finite=0;double vmin=INFINITY,vmax=-INFINITY;
 for(long long off=0;off<np;off+=CH){size_t nn=(size_t)std::min<long long>(CH,np-off);CHECK_CUDA(cudaMemcpy(buf.data(),C+off,nn*sizeof(float),cudaMemcpyDeviceToHost));SHA256_Update(&sh,buf.data(),nn*sizeof(float));for(size_t j=0;j<nn;j++)if(std::isfinite(buf[j])){finite++;vmin=std::min(vmin,(double)buf[j]);vmax=std::max(vmax,(double)buf[j]);}}
 unsigned char digest[SHA256_DIGEST_LENGTH];SHA256_Final(digest,&sh);double scan_s=elapsed(tscan);
 std::ofstream fs(out+"/full_stats.json");fs<<std::setprecision(17)<<"{\"ordering\":\"NEST\",\"rawN\":{\"dtype\":\"float32\",\"count\":"<<np<<",\"finite_count\":"<<finite<<",\"min\":"<<vmin<<",\"max\":"<<vmax<<",\"sha256\":\""<<hex_sha(digest)<<"\"},\"scan_wall_s\":"<<scan_s<<",\"full_map_written\":false}\n";
 std::ofstream seg(out+"/segments.csv");seg<<"segment,h2d_event_s,prep_wall_s,recon_wall_s,recon_stream_span_s,segment_wall_s,vv_tasks,mixed_tasks\n"<<std::setprecision(17);for(auto&t:timings)seg<<t.sid<<","<<t.h2d<<","<<t.prep<<","<<t.recon<<","<<t.span<<","<<t.wall<<","<<t.vv<<","<<t.mixed<<"\n";
 std::ofstream meta(out+"/metrics.json");meta<<std::setprecision(17)<<"{\"method\":\""<<(LEGACY_FASTPATH?"legacy_combined_fastpath":"controlled_fixed_direct")<<"\",\"nside\":4096,\"day\":1,\"segments\":10,\"output_pixels\":"<<np<<",\"half_baselines\":1954680,\"cache_load_decode_wall_s\":"<<cache_s<<",\"init_wall_s\":"<<init_s<<",\"inverse_loop_wall_s\":"<<loop_s<<",\"output_finalize_wall_s\":0,\"sample_export_wall_s\":"<<sample_s<<",\"full_scan_wall_s\":"<<scan_s<<",\"estimated_min_allocation_bytes\":"<<estimated_min<<",\"plan_allocated_bytes\":"<<plan_bytes<<",\"observed_init_device_occupied_bytes\":"<<(totalmem-free_init)<<",\"observed_boundary_max_device_occupied_bytes\":"<<(totalmem-minfree)<<",\"peak_memory_method\":\"cudaMemGetInfo at init and segment boundaries; not continuous peak\",\"sameV\":\"cached G1 consumed-n production visibility\",\"output\":\"raw weighted N; diagnostic W is external previously validated, no full-W computation in this loop\",\"normalization\":\"no hidden D\",\"forward_executed\":false,\"full_map_written\":false}\n";
 if(finite!=np)throw std::runtime_error("nonfinite full output");std::cout<<"FULL_INVERSE_DONE "<<(LEGACY_FASTPATH?"legacy_combined_fastpath":"controlled_fixed_direct")<<" loop="<<loop_s<<"\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 2;}}
