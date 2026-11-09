#if ATOMIC_TASKLIST
#include "../source/recon_fastpath_atomic/common.hpp"
#include "../source/recon_fastpath_atomic/orbit_online.cuh"
#include "../source/recon_fastpath_atomic/viss_recon_kernel.cuh"
#elif LOOKUP_FIXED
#include "../source/recon_fastpath_fixed_index/common.hpp"
#include "../source/recon_fastpath_fixed_index/orbit_online.cuh"
#include "../source/recon_fastpath_fixed_index/viss_recon_kernel.cuh"
#else
#include "../source/recon_fastpath/common.hpp"
#include "../source/recon_fastpath/orbit_online.cuh"
#include "../source/recon_fastpath/viss_recon_kernel.cuh"
#endif
#include <array>
#include <stdexcept>

struct CacheHeader{char magic[8];int32_t nside,day,seg,t0,tlen,nhalf,G;};
struct Row{float u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im;};
static_assert(sizeof(CacheHeader)==36 && sizeof(Row)==60,"frozen replay format");
template<class T>static std::vector<T> load(const std::string& p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto bytes=f.tellg();if(bytes<0||(size_t)bytes%sizeof(T))throw std::runtime_error("shape: "+p);std::vector<T> v((size_t)bytes/sizeof(T));f.seekg(0);f.read((char*)v.data(),bytes);if(!f.good())throw std::runtime_error("read: "+p);return v;}
template<class T>static void write(const std::string& p,const std::vector<T>& v){std::ofstream f(p,std::ios::binary);f.write((char*)v.data(),v.size()*sizeof(T));if(!f.good())throw std::runtime_error("write: "+p);}
template<class T>static T* device(const std::vector<T>& v){T* d=nullptr;CHECK_CUDA(cudaMalloc(&d,v.size()*sizeof(T)));CHECK_CUDA(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>static std::vector<T> host(const T* d,size_t n){std::vector<T> v(n);CHECK_CUDA(cudaMemcpy(v.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}

__global__ void mask_probe(const float* l,const float* m,const float* n,const float* x1,const float* y1,const float* z1,const float* x2,const float* y2,const float* z2,int nb,float cp,unsigned char* out){
 int j=blockIdx.x*blockDim.x+threadIdx.x;if(j>=16*nb)return;int p=j/nb,h=j%nb;
 out[j]=operator_point_visible_pair_dirs(x1[h],y1[h],z1[h],x2[h],y2[h],z2[h],l[p],m[p],n[p],cp);
}

__global__ void weighted_visible_sum(const float* l,const float* m,const float* n,int np,const float* x1,const float* y1,const float* z1,const float* x2,const float* y2,const float* z2,int nb,float cp,const float* q,double* W){
 int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=np)return;double sum=0;
 for(int h=0;h<nb;h++)if(operator_point_visible_pair_dirs(x1[h],y1[h],z1[h],x2[h],y2[h],z2[h],l[p],m[p],n[p],cp))sum+=(double)q[h];
 W[p]+=sum;
}
static float f32dot(const std::array<float,3>& p,const std::array<float,3>& x,int variant){
 static const int order[6][3]={{0,1,2},{0,2,1},{1,0,2},{1,2,0},{2,0,1},{2,1,0}};
 const int* q=order[variant];float a=p[q[0]]*x[q[0]];a=std::fma(p[q[1]],x[q[1]],a);return std::fma(p[q[2]],x[q[2]],a);
}

int main(int argc,char** argv){try{
 std::string gpu=get_arg(argc,argv,"--gpus","3");setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID",1);setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);CHECK_CUDA(cudaSetDevice(0));
 bool compute_W=std::stoi(get_arg(argc,argv,"--compute_W","1"))!=0;
 int max_segments=std::stoi(get_arg(argc,argv,"--max_segments","1"));if(max_segments<1||max_segments>10)throw std::runtime_error("max_segments must be 1..10");
 std::string replay_path=get_arg(argc,argv,"--replay_path","tasklist_single");if(replay_path!="tasklist_single"&&replay_path!="tasklist_parallel"&&replay_path!="direct")throw std::runtime_error("replay_path must be tasklist_single|tasklist_parallel|direct");
 std::string cache=get_arg(argc,argv,"--cache_root",""),input=get_arg(argc,argv,"--case_dir",""),out=get_arg(argc,argv,"--out_dir",""),dcfpath=get_arg(argc,argv,"--dcf_bin","");
 if(cache.empty()||input.empty()||out.empty()||dcfpath.empty())throw std::runtime_error("paths required");
 auto ids=load<uint32_t>(input+"/ids.u32");auto l=load<float>(input+"/l.f32"),m=load<float>(input+"/m.f32"),n=load<float>(input+"/n.f32");int np=l.size();if(np!=8192||ids.size()!=8192||m.size()!=l.size()||n.size()!=l.size())throw std::runtime_error("8192 fixed directions required");
 DcfMbBinPayload payload;if(!load_dcf_mb_bin(dcfpath,payload))throw std::runtime_error("DCF load");auto dcf=device(payload.dcf);auto dl=device(l),dm=device(m),dn=device(n);
 std::vector<uint32_t> probes(16);std::vector<float> pl(16),pm(16),pn(16);for(int j=0;j<16;j++){probes[j]=(uint32_t)((j+.5)*np/16);pl[j]=l[probes[j]];pm[j]=m[probes[j]];pn[j]=n[probes[j]];}
 auto dpl=device(pl),dpm=device(pm),dpn=device(pn);
 // Sentinel metadata forces every original classifier to [-1,+1], so no
 // wide sampled tile ever takes a cone shortcut. The original kernel is untouched.
 std::vector<float> zero(np/256,0),one(np/256,1);auto dzero=device(zero),done=device(one);
 float* C[2];for(int mode=0;mode<2;mode++){CHECK_CUDA(cudaMalloc(&C[mode],np*sizeof(float)));CHECK_CUDA(cudaMemset(C[mode],0,np*sizeof(float)));}
 double* Wdev[2]={};if(compute_W)for(int k=0;k<2;k++){CHECK_CUDA(cudaMalloc(&Wdev[k],np*sizeof(double)));CHECK_CUDA(cudaMemset(Wdev[k],0,np*sizeof(double)));}
 long double cpuW[7][2][16]={};
 long double cpu[7][2][16]={},ambiguous_bound[2][16]={};uint32_t counts[7][16]={},gpu_counts[16]={},ambiguous_counts[16]={};uint64_t outside_bound=0;double max_norm_deviation=0;long double qsum[2]={},q2sum[2]={};uint64_t totalrows=0;
 float R=1737.1e3f,H=300e3f;float theta=asinf(R/(R+H));float cp=cosf((float)M_PI-theta);
 std::ofstream weights_out(out+"/pair_weights_legacy.f32",std::ios::binary);
 std::ofstream segments(out+"/segments.csv");segments<<"segment,half_baselines,cosphi\n";
 for(int sid=1;sid<=max_segments;sid++){
  std::ifstream file(cache+"/day1_seg"+std::to_string(sid)+".bin",std::ios::binary);CacheHeader hdr{};file.read((char*)&hdr,sizeof(hdr));if(!file.good()||std::memcmp(hdr.magic,"PREPLAY1",8)||hdr.nside!=4096||hdr.day!=1||hdr.seg!=sid||hdr.nhalf!=195468||hdr.G!=1)throw std::runtime_error("cache identity");std::vector<Row> rows(hdr.nhalf);file.read((char*)rows.data(),rows.size()*sizeof(Row));if(!file.good())throw std::runtime_error("cache read");int nb=rows.size();totalrows+=nb;
  std::array<std::vector<float>,9> a;for(auto& v:a)v.resize(nb);std::vector<float2> vis(nb);for(int h=0;h<nb;h++){auto r=rows[h];float p[9]={r.u,r.v,r.w,r.x1,r.y1,r.z1,r.x2,r.y2,r.z2};for(int j=0;j<9;j++)a[j][h]=p[j];vis[h]=make_float2(r.re,r.im);}
  std::array<float*,9>d;for(int j=0;j<9;j++)d[j]=device(a[j]);auto dv=device(vis);
  normalize3_inplace_kernel<<<(nb+255)/256,256>>>(d[3],d[4],d[5],nb);normalize3_inplace_kernel<<<(nb+255)/256,256>>>(d[6],d[7],d[8],nb);
  float* q[2];for(int k=0;k<2;k++)CHECK_CUDA(cudaMalloc(&q[k],nb*sizeof(float)));std::vector<float> unit(nb,2);CHECK_CUDA(cudaMemcpy(q[0],unit.data(),nb*sizeof(float),cudaMemcpyHostToDevice));
  compute_pair_weight_half_kernel<<<(nb+255)/256,256>>>(d[0],d[1],d[2],nb,dcf,payload.dcf.size(),q[1]);
  if(compute_W)for(int mode=0;mode<2;mode++)weighted_visible_sum<<<(np+255)/256,256>>>(dl,dm,dn,np,d[3],d[4],d[5],d[6],d[7],d[8],nb,cp,q[mode],Wdev[mode]);
  for(int mode=0;mode<2;mode++){
   // Include the frozen legacy production header and use its preweight kernel.
   // All sampled tiles have sentinel cones [-1,+1]; every task is mixed.
   CHECK_CUDA(cudaMemcpy(dv,vis.data(),nb*sizeof(float2),cudaMemcpyHostToDevice));
   apply_pair_weight_to_viss_half_kernel<<<(nb+255)/256,256>>>(dv,q[mode],nb);
   if(replay_path=="direct"){
    recon_3d_direct_halfsym_tilecone_real<256,128,true><<<np/256,256>>>(np,dl,dm,dn,dzero,dzero,dzero,dzero,done,np/256,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],dv,nb,cp,C[mode]);
   }else{
    // Explicit mixed IDs bypass only host/CUB planning, not the recon math.
    // The single path gives one writer/pixel; the parallel path preserves
    // original 512-baseline chunks and is a separate concurrency diagnostic.
    // This is a numerical gate, deliberately not a planner/performance test.
    const int dispatch_baselines=(replay_path=="tasklist_parallel"?512:128);
    for(int b0=0;b0<nb;b0+=dispatch_baselines){
     int cn=std::min(dispatch_baselines,nb-b0),nt=(cn+127)/128*(np/256);
     std::vector<int> ids_task(nt);for(int j=0;j<nt;j++)ids_task[j]=j;
     auto dt=device(ids_task);
     recon_3d_direct_tasklist_mixed_real<256,128><<<nt,256>>>(np,dl,dm,dn,dzero,dzero,dzero,dzero,done,np/256,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],dv,b0,cn,cp,dt,nt,C[mode]);
     CHECK_CUDA(cudaFree(dt));
    }
   }
  }
  unsigned char* masks=nullptr;CHECK_CUDA(cudaMalloc(&masks,(size_t)16*nb));mask_probe<<<(16*nb+255)/256,256>>>(dpl,dpm,dpn,d[3],d[4],d[5],d[6],d[7],d[8],nb,cp,masks);CHECK_CUDA(cudaPeekAtLastError());auto gpu_mask=host(masks,(size_t)16*nb);
  for(int j=3;j<9;j++)a[j]=host(d[j],nb);std::array<std::vector<float>,2> weight;for(int k=0;k<2;k++)weight[k]=host(q[k],nb);
  weights_out.write((char*)weight[1].data(),nb*sizeof(float));
  for(int h=0;h<nb;h++){
   for(int e=0;e<2;e++){double norm=std::sqrt((double)a[3+3*e][h]*a[3+3*e][h]+(double)a[4+3*e][h]*a[4+3*e][h]+(double)a[5+3*e][h]*a[5+3*e][h]);max_norm_deviation=std::max(max_norm_deviation,std::fabs(norm-1));}
   for(int k=0;k<2;k++){float v=weight[k][h];if(!std::isfinite(v)||v<0||(k==0?v!=2:v>.25f))throw std::runtime_error("weight range/factor");qsum[k]+=v;q2sum[k]+=(long double)v*v;}
  }
  #pragma omp parallel for num_threads(8) reduction(+:outside_bound)
  for(int j=0;j<16;j++){
   std::array<float,3> p={pl[j],pm[j],pn[j]};
   for(int h=0;h<nb;h++){
    auto r=rows[h];std::array<float,3>x={a[3][h],a[4][h],a[5][h]},y={a[6][h],a[7][h],a[8][h]};double dx=(double)p[0]*x[0]+(double)p[1]*x[1]+(double)p[2]*x[2],dy=(double)p[0]*y[0]+(double)p[1]*y[1]+(double)p[2]*y[2];bool physical=dx>=cp&&dy>=cp;
    double phase=2*M_PI*((double)r.u*p[0]+(double)r.v*p[1]+(double)r.w*p[2]);double value=(double)r.re*std::cos(phase)-(double)r.im*std::sin(phase);
    // gamma5 bound covers any two-FMA/one-product FP32 association.
    double gamma=5*0x1p-24/(1-5*0x1p-24);double ex=gamma*(std::fabs((double)p[0]*x[0])+std::fabs((double)p[1]*x[1])+std::fabs((double)p[2]*x[2]));double ey=gamma*(std::fabs((double)p[0]*y[0])+std::fabs((double)p[1]*y[1])+std::fabs((double)p[2]*y[2]));bool ambiguous=std::fabs(dx-cp)<=ex||std::fabs(dy-cp)<=ey;
    bool broad_ambiguous=std::fabs(dx-cp)<=2e-6||std::fabs(dy-cp)<=2e-6;
    bool actual=gpu_mask[(size_t)j*nb+h]!=0;gpu_counts[j]+=actual;ambiguous_counts[j]+=broad_ambiguous;if(actual!=physical&&!ambiguous)outside_bound++;
    for(int k=0;k<2;k++){if(physical){cpu[6][k][j]+=(long double)weight[k][h]*value;cpuW[6][k][j]+=(long double)weight[k][h];}if(broad_ambiguous)ambiguous_bound[k][j]+=(long double)weight[k][h]*std::hypot((double)r.re,(double)r.im);}
    counts[6][j]+=physical;
    for(int v=0;v<6;v++){bool mask=f32dot(p,x,v)>=cp&&f32dot(p,y,v)>=cp;counts[v][j]+=mask;if(mask)for(int k=0;k<2;k++){cpu[v][k][j]+=(long double)weight[k][h]*value;cpuW[v][k][j]+=(long double)weight[k][h];}}
   }
  }
  for(auto ptr:d)CHECK_CUDA(cudaFree(ptr));for(auto ptr:q)CHECK_CUDA(cudaFree(ptr));CHECK_CUDA(cudaFree(dv));CHECK_CUDA(cudaFree(masks));segments<<sid<<","<<nb<<","<<std::setprecision(17)<<cp<<"\n";segments.flush();std::cout<<"LEGACY_FASTPATH_SEG_DONE "<<sid<<"/10\n";std::cout.flush();
 }
 for(int mode=0;mode<2;mode++)write(out+"/raw_N_mode"+std::to_string(mode)+".f32",host(C[mode],np));if(compute_W)for(int mode=0;mode<2;mode++)write(out+"/W_mode"+std::to_string(mode)+".f64",host(Wdev[mode],np));write(out+"/ids.u32",ids);write(out+"/probe_sample_indices.u32",probes);
 std::ofstream oracle(out+"/cpu_oracle.csv");oracle<<"variant,mode,probe,count,N,W,ambiguity_absolute_bound,gpu_aux_count,ambiguous_count\n"<<std::setprecision(20);for(int v=0;v<7;v++)for(int k=0;k<2;k++)for(int j=0;j<16;j++)oracle<<v<<","<<k<<","<<j<<","<<counts[v][j]<<","<<(double)cpu[v][k][j]<<","<<(double)cpuW[v][k][j]<<","<<(double)ambiguous_bound[k][j]<<","<<gpu_counts[j]<<","<<ambiguous_counts[j]<<"\n";
 std::ofstream meta(out+"/execution_metrics.json");meta<<std::setprecision(17)<<"{\"lookup_fixed\":"<<LOOKUP_FIXED<<",\"rows\":"<<totalrows<<",\"cosphi\":"<<cp<<",\"max_unit_norm_deviation\":"<<max_norm_deviation<<",\"point_mask_disagreements_outside_proved_bound\":"<<outside_bound<<",\"weight_sum\":["<<(double)qsum[0]<<","<<(double)qsum[1]<<"]}\n";

 std::cout<<"LEGACY_FASTPATH_REPLAY_DONE\n";return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<"\n";return 2;}}
