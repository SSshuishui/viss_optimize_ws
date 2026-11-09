#pragma once
#include <array>
#include <stdexcept>
#include <sys/statvfs.h>

struct ReplayRow { float u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im; };
static_assert(sizeof(ReplayRow)==60,"cache row layout");
struct ReplayHeader { char magic[8]; int32_t nside,day,seg,t0,tlen,nhalf,forward_gpus; };
static_assert(sizeof(ReplayHeader)==36,"cache header layout");
static uint64_t splitmix64(uint64_t x){
  x+=0x9e3779b97f4a7c15ULL;x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
  x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
static std::vector<uint32_t> sample_ids(int nside,int count){
  uint64_t np=12ULL*nside*nside;std::vector<uint32_t> ids(count);
  for(int j=0;j<count;j++){
    uint64_t lo=(uint64_t)j*np/count,hi=(uint64_t)(j+1)*np/count;
    ids[j]=(uint32_t)(lo+splitmix64(20261101ULL+j)%(hi-lo));
  }return ids;
}
static void disk_guard(const std::string& root,size_t required=1024ULL*1024*1024){
  struct statvfs st{};if(statvfs(root.c_str(),&st))throw std::runtime_error("statvfs failed");
  if((uint64_t)st.f_bavail*st.f_frsize<required)throw std::runtime_error("disk guard: less than required free bytes");
}
template<typename T> static T* device_copy(const std::vector<T>& host){
  T* ptr=nullptr;CHECK_CUDA(cudaMalloc(&ptr,host.size()*sizeof(T)));
  CHECK_CUDA(cudaMemcpy(ptr,host.data(),host.size()*sizeof(T),cudaMemcpyHostToDevice));return ptr;
}
template<typename T> static std::vector<T> host_copy(const T* ptr,size_t n){
  std::vector<T> h(n);CHECK_CUDA(cudaMemcpy(h.data(),ptr,n*sizeof(T),cudaMemcpyDeviceToHost));return h;
}
template<typename T> static void write_values(const std::string& path,const std::vector<T>& values){
  std::ofstream o(path,std::ios::binary);o.write((const char*)values.data(),values.size()*sizeof(T));
  if(!o.good())throw std::runtime_error("write failed "+path);
}
__global__ void gather_samples(const uint32_t* ids,int ns,long long start,long long end,
                               const float* in,float* out){
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j<ns && ids[j]>=(uint64_t)start && ids[j]<(uint64_t)end)out[j]=in[ids[j]-start];
}
__global__ void gather_samples_u32(const uint32_t* ids,int ns,long long start,long long end,
                                   const uint32_t* in,uint32_t* out){
  int j=blockIdx.x*blockDim.x+threadIdx.x;
  if(j<ns && ids[j]>=(uint64_t)start && ids[j]<(uint64_t)end)out[j]=in[ids[j]-start];
}
static void save_cache(const std::string& root,int nside,int day,int seg,int ngpu,
                       const OrbitSegSlot& src,const GpuSegSlot& slot,
                       const std::vector<float2>& raw,const std::vector<float2>& corrected){
  disk_guard(root);CHECK_CUDA(cudaStreamSynchronize(0));
  auto inv1=host_copy(slot.d_invn1,src.segN),inv2=host_copy(slot.d_invn2,src.segN);
  std::vector<ReplayRow> rows(corrected.size());
  for(size_t h=0;h<rows.size();h++){
    int f=half_to_full_pos_idx((int)h);
    rows[h]={src.h_u[f],src.h_v[f],src.h_w[f],src.h_x1[f],src.h_y1[f],src.h_z1[f],inv1[f],
             src.h_x2[f],src.h_y2[f],src.h_z2[f],inv2[f],raw[h].x,raw[h].y,corrected[h].x,corrected[h].y};
  }
  ReplayHeader hdr{{'P','R','E','P','L','A','Y','1'},nside,day,seg,src.t0,src.tlen,(int)rows.size(),ngpu};
  std::string path=root+"/day"+std::to_string(day)+"_seg"+std::to_string(seg)+".bin";
  std::ofstream o(path+".tmp",std::ios::binary);o.write((char*)&hdr,sizeof(hdr));
  o.write((char*)rows.data(),rows.size()*sizeof(ReplayRow));o.close();
  if(!o.good() || std::rename((path+".tmp").c_str(),path.c_str()))throw std::runtime_error("cache write failed");
  std::cout<<"REPLAY_CACHE "<<path<<" stage=host_reduced_raw_nm1_and_host_phase_corrected_n nhalf="<<rows.size()<<"\n";
}

