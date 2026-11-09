#include "../source/weighted_diagnostic/common.hpp"
#include "../source/weighted_diagnostic/orbit_online.cuh"
#include "../source/weighted_diagnostic/viss_recon_kernel.cuh"
#include <array>
#include <stdexcept>
struct CacheHeader{char magic[8];int32_t nside,day,seg,t0,tlen,nhalf,G;};
struct Row{float u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im;};
static_assert(sizeof(CacheHeader)==36&&sizeof(Row)==60,"cache format");
template<class T>static std::vector<T>readv(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);auto n=f.tellg();if(n<0||(size_t)n%sizeof(T))throw std::runtime_error("shape");std::vector<T>v((size_t)n/sizeof(T));f.seekg(0);f.read((char*)v.data(),n);if(!f.good())throw std::runtime_error("read");return v;}
template<class T>static T*dev(const std::vector<T>&v){T*d=nullptr;CHECK_CUDA(cudaMalloc(&d,v.size()*sizeof(T)));CHECK_CUDA(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));return d;}
template<class T>static void output(const std::string&p,T*d,int n){std::vector<T>h(n);CHECK_CUDA(cudaMemcpy(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost));std::ofstream f(p,std::ios::binary);f.write((char*)h.data(),n*sizeof(T));if(!f.good())throw std::runtime_error("write");}
int main(int argc,char**argv){try{
 const std::string gpu=get_arg(argc,argv,"--gpus","3"),cache=get_arg(argc,argv,"--cache_root",""),sample=get_arg(argc,argv,"--sample_dir",""),out=get_arg(argc,argv,"--out_dir",""),mask=get_arg(argc,argv,"--mask","point");
 if(cache.empty()||sample.empty()||out.empty()||(mask!="point"&&mask!="all_visible"))throw std::runtime_error("arguments");setenv("CUDA_DEVICE_ORDER","PCI_BUS_ID",1);setenv("CUDA_VISIBLE_DEVICES",gpu.c_str(),1);CHECK_CUDA(cudaSetDevice(0));
 auto ids=readv<uint32_t>(sample+"/ids.u32");auto l=readv<float>(sample+"/l.f32"),m=readv<float>(sample+"/m.f32"),n=readv<float>(sample+"/n.f32");if(ids.size()!=8192||l.size()!=8192||m.size()!=8192||n.size()!=8192)throw std::runtime_error("8192 directions");for(int j=0;j<8192;j++)if(ids[j]>=201326592u||(j&&ids[j]<=ids[j-1]))throw std::runtime_error("sample ID range/strict ordering");
 auto dl=dev(l),dm=dev(m),dn=dev(n);std::vector<float>zero(32,0),one(32,1);auto dz=dev(zero),do_=dev(one);
 std::ifstream f(cache+"/day1_seg1.bin",std::ios::binary);CacheHeader h{};f.read((char*)&h,sizeof(h));if(!f.good()||std::memcmp(h.magic,"PREPLAY1",8)||h.nside!=4096||h.day!=1||h.seg!=1||h.nhalf!=195468||h.t0!=0||h.tlen!=6981||h.nhalf!=28*h.tlen||h.G!=1)throw std::runtime_error("cache identity");std::vector<Row>rows(h.nhalf);f.read((char*)rows.data(),rows.size()*sizeof(Row));if(!f.good())throw std::runtime_error("cache body");
 const int nb=rows.size();std::array<std::vector<float>,9>a;for(auto&v:a)v.resize(nb);std::vector<float2>V(nb);for(int j=0;j<nb;j++){auto r=rows[j];float v[9]={r.u,r.v,r.w,r.x1,r.y1,r.z1,r.x2,r.y2,r.z2};for(int k=0;k<9;k++)a[k][j]=v[k];V[j]=make_float2(r.re,r.im);}
 std::array<float*,9>d;for(int k=0;k<9;k++)d[k]=dev(a[k]);auto dv=dev(V);normalize3_inplace_kernel<<<(nb+255)/256,256>>>(d[3],d[4],d[5],nb);normalize3_inplace_kernel<<<(nb+255)/256,256>>>(d[6],d[7],d[8],nb);
 DcfMbBinPayload p;if(!load_dcf_mb_bin("/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/dcf_mb_10M_days450_seed42.bin",p))throw std::runtime_error("DCF");auto dcf=dev(p.dcf);float*q=nullptr;CHECK_CUDA(cudaMalloc(&q,nb*sizeof(float)));compute_pair_weight_half_kernel<<<(nb+255)/256,256>>>(d[0],d[1],d[2],nb,dcf,p.dcf.size(),q);
 float*C=nullptr;double*W=nullptr;CHECK_CUDA(cudaMalloc(&C,8192*sizeof(float)));CHECK_CUDA(cudaMalloc(&W,8192*sizeof(double)));float cp=cosf((float)M_PI-asinf(1737.1e3f/(1737.1e3f+300e3f)));
 for(int variant=0;variant<2;variant++){
  CHECK_CUDA(cudaMemset(C,0,8192*sizeof(float)));CHECK_CUDA(cudaMemset(W,0,8192*sizeof(double)));if(variant==0)CHECK_CUDA(cudaMemcpy(dv,V.data(),nb*sizeof(float2),cudaMemcpyHostToDevice));else CHECK_CUDA(cudaMemset(dv,0,nb*sizeof(float2)));
  apply_pair_weight_to_viss_half_kernel<<<(nb+255)/256,256>>>(dv,q,nb);
  for(int b0=0;b0<nb;b0+=512){int cn=std::min(512,nb-b0),ntasks=(cn+127)/128*32;std::vector<int>tasks(ntasks);for(int j=0;j<ntasks;j++)tasks[j]=j;auto dt=dev(tasks);
   if(mask=="point")recon_3d_direct_tasklist_mixed_real<256,128><<<ntasks,256>>>(8192,dl,dm,dn,dz,dz,dz,dz,do_,32,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],dv,b0,cn,cp,dt,ntasks,C,q,W);
   else recon_3d_direct_tasklist_vv_real<256,128><<<ntasks,256>>>(8192,dl,dm,dn,32,d[0],d[1],d[2],dv,b0,cn,dt,ntasks,C,q,W);
   CHECK_CUDA(cudaPeekAtLastError());CHECK_CUDA(cudaFree(dt));
  }
  CHECK_CUDA(cudaDeviceSynchronize());output(out+"/N_"+(variant==0?"real":"zero")+".f32",C,8192);output(out+"/W_"+(variant==0?"real":"zero")+".f64",W,8192);
 }
 std::vector<float>qh(nb);CHECK_CUDA(cudaMemcpy(qh.data(),q,nb*sizeof(float),cudaMemcpyDeviceToHost));long double sum=0;for(float v:qh)sum+=v;
 std::ofstream meta(out+"/control.json");meta<<std::setprecision(20)<<"{\"mask_mode\":\""<<mask<<"\",\"points\":8192,\"half_baselines\":"<<nb<<",\"q_sum_independent_CPU\":"<<(double)sum<<",\"OP_RECON_SKIP_ZERO_WEIGHTED_VISS\":"<<OP_RECON_SKIP_ZERO_WEIGHTED_VISS<<",\"OP_RECON_DEBUG_CHECK_FINITE\":"<<OP_RECON_DEBUG_CHECK_FINITE<<",\"purpose\":\"W must be independent of real/zero V; all_visible separately exercises uniformW reduction\"}\n";
 std::cout<<"ZERO_V_CONTROL_DONE\n";return 0;
}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<"\n";return 2;}}
