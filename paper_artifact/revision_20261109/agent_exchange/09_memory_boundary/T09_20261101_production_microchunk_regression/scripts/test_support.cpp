// CPU-only host-helper test: CUDA types/APIs are stubs, no device runtime linked.
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
struct float2{float x,y;};struct double2{double x,y;};
inline float2 make_float2(float a,float b){return {a,b};}
struct Dim{int x=0;};inline Dim blockIdx,blockDim,threadIdx;
#define __global__
using cudaError_t=int;constexpr int cudaSuccess=0,cudaMemcpyDeviceToHost=0;
inline int cudaSetDevice(int){return 0;}inline int cudaDeviceSynchronize(){return 0;}
inline int cudaMemcpy(void*d,const void*s,size_t n,int){std::memcpy(d,s,n);return 0;}
#define CHECK_CUDA(x) do{if(x)throw std::runtime_error("stub");}while(0)
struct GpuCtx{float*d_Cacc=nullptr;uint32_t*d_Wacc=nullptr;long long n_chunk=0,pix0=0;};
struct OrbitSegSlot{int t0=0,tlen=0,segN=0;float*h_u=nullptr,*h_v=nullptr,*h_w=nullptr,*h_x1=nullptr,*h_y1=nullptr,*h_z1=nullptr,*h_x2=nullptr,*h_y2=nullptr,*h_z2=nullptr;};
struct GroupArtifactsHost{int nuniq=0;std::vector<int>keys_unique,rep_half_idx;};
namespace prodalloc{inline std::atomic<size_t>peak[2],minfree[2];inline void observe(int,size_t,int){}}
#include "../source/regression_support.hpp"
template<class F>void reject(F f){bool yes=false;try{f();}catch(const std::exception&){yes=true;}if(!yes)throw std::runtime_error("negative host fixture accepted");}
int main(int argc,char**argv){namespace fs=std::filesystem;fs::path dir(argv[1]);fs::create_directories(dir);auto c=prod::forward_identity(std::string(64,'a'),4,2,1<<27,123);std::vector<float2>v(586432,{1,2});prod::commit_forward(dir,c,v);auto r=prod::load_forward(dir,c);if(std::memcmp(v.data(),r.data(),v.size()*8))return 2;auto wrong=c;wrong.mode=0;reject([&]{prod::load_forward(dir,wrong);});wrong=c;wrong.G=3;reject([&]{prod::load_forward(dir,wrong);});wrong=c;wrong.geometry++;reject([&]{prod::load_forward(dir,wrong);});wrong=c;wrong.slab=1<<25;reject([&]{prod::load_forward(dir,wrong);});wrong=c;wrong.identity[0]='b';reject([&]{prod::load_forward(dir,wrong);});{std::ofstream f(dir/"forward_raw.f32x2",std::ios::binary|std::ios::app);f.put('x');}reject([&]{prod::load_forward(dir,c);});prod::commit_forward(dir,c,v);
 std::string a="checkpoint0",b="checkpoint1";prod::text(dir/"forward_gpu0.ck",a);prod::text(dir/"forward_gpu1.ck",b);prod::checkpoint_cleanup(dir,2);if(fs::exists(dir/"forward_gpu0.ck"))return 3;auto before=prod::file_sha(dir/"checkpoint_cleanup.json");prod::text(dir/"forward_gpu0.ck",a);prod::checkpoint_cleanup(dir,2);if(before!=prod::file_sha(dir/"checkpoint_cleanup.json")||fs::exists(dir/"forward_gpu0.ck"))return 4;prod::text(dir/"forward_gpu0.ck","changed");reject([&]{prod::checkpoint_cleanup(dir,2);});prodalloc::minfree[0]=1;prodalloc::minfree[1]=1;reject([&]{prod::memory(dir,2,true);});std::ifstream f(dir/"memory.json");std::string s((std::istreambuf_iterator<char>(f)),{});if(s.find("\"pass\":false")==std::string::npos)return 5;fs::remove_all(dir);std::cout<<"10 host commit/identity/SHA/EOF/cleanup-resume/memory-negative fixtures PASS\n";}
