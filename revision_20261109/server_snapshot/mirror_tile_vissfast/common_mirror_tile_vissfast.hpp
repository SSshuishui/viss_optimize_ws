#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <random>
#include <iomanip>

#include <cuda_runtime.h>
#include <omp.h>

#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call) do { \
  cudaError_t __cuda_err = (call); \
  if(__cuda_err != cudaSuccess){ \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(__cuda_err)); \
    std::exit(1); \
  } \
} while(0)

#ifndef SINCOS_FAST_DEFINED
#define SINCOS_FAST_DEFINED
__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) {
  __sincosf(x, s, c);
}
#endif

static constexpr int SIGNED_BASELINES_PER_T = 56;
static constexpr int UNIQUE_BASELINES_PER_T = 28;

struct HostTimer {
  using clock=std::chrono::high_resolution_clock;
  clock::time_point t0;
  void tic(){t0=clock::now();}
  double toc_s() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now()-t0).count();
  }
};

static inline std::string get_arg(int argc,char** argv,const std::string& key,const std::string& defv){
  for(int i=1;i<argc;i++){
    std::string s(argv[i]);
    if(s.rfind(key+"=",0)==0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline int to_int(const std::string& s,int defv){ try{return std::stoi(s);}catch(...){return defv;} }
static inline long long to_ll(const std::string& s,long long defv){ try{return std::stoll(s);}catch(...){return defv;} }
static inline float to_float(const std::string& s,float defv){ try{return std::stof(s);}catch(...){return defv;} }

static inline std::vector<int> parse_gpus(const std::string& s){
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while(std::getline(ss,tok,',')) if(!tok.empty()) out.push_back(std::stoi(tok));
  if(out.empty()) out.push_back(0);
  return out;
}

static inline std::string norm_dir(std::string p){
  while(!p.empty() && p.back()=='/') p.pop_back();
  return p;
}
static inline void ensure_dir(const std::string& path){
  std::string cmd = "mkdir -p " + path;
  int rc = std::system(cmd.c_str());
  (void)rc;
}

// -------- robust loader: file may be headerless or first line is npix --------
static bool load_single_auto_np(const std::string& path, float* out, long long n_expected) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  char* p = line;
  float first = strtof(p, &p);
  long long i = 0;

  long long maybeN = (long long)llround((double)first);
  bool is_header = (std::llabs(maybeN - n_expected) == 0);

  if (!is_header) out[i++] = first;

  while (i < n_expected && fgets(line, sizeof(line), fp)) {
    char* q = line;
    out[i++] = strtof(q, &q);
  }
  fclose(fp);
  return (i == n_expected);
}

static bool load_single_bin_exact(const std::string& path, float* out, long long n_expected) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs.is_open()) return false;
  ifs.read(reinterpret_cast<char*>(out), (std::streamsize)(n_expected * (long long)sizeof(float)));
  std::streamsize got = ifs.gcount();
  return got == (std::streamsize)(n_expected * (long long)sizeof(float));
}

static bool load_B_auto(const std::string& sky_dir,
                        const std::string& btag,
                        float* out,
                        long long npix,
                        const std::string& mode,
                        std::string* used_path = nullptr) {
  auto try_txt = [&](const std::string& p)->bool{
    if (load_single_auto_np(p, out, npix)) { if(used_path) *used_path = p; return true; }
    return false;
  };
  auto try_bin = [&](const std::string& p)->bool{
    if (load_single_bin_exact(p, out, npix)) { if(used_path) *used_path = p; return true; }
    return false;
  };

  const std::string b1 = sky_dir + "/B_10M" + ".bin";
  const std::string b2 = sky_dir + "/B.bin";
  const std::string t1 = sky_dir + "/B_10M" + ".txt";
  const std::string t2 = sky_dir + "/B.txt";

  if (mode == "bin") return try_bin(b1) || try_bin(b2);
  if (mode == "txt") return try_txt(t1) || try_txt(t2);
  return try_bin(b1) || try_bin(b2) || try_txt(t1) || try_txt(t2);
}

static void write_txt_3cols_stream(std::ofstream& ofs,
                                   const float* a, const float* b, const float* c,
                                   size_t n) {
  ofs.setf(std::ios::scientific);
  ofs << std::setprecision(15);
  for (size_t i = 0; i < n; ++i) {
    ofs << (double)a[i] << " " << (double)b[i] << " " << (double)c[i] << "\n";
  }
}

static inline uint64_t to_u64(const std::string& s, uint64_t defv) {
  try { return (uint64_t)std::stoull(s); } catch(...) { return defv; }
}

// 3-col loader, skip first line
static bool load_triplets_skip_first(const std::string& path,float* a,float* b,float* c,int maxN,int& outN){
  FILE* fp=fopen(path.c_str(),"r");
  if(!fp) return false;
  char line[512];
  if(!fgets(line,sizeof(line),fp)){ fclose(fp); return false; }
  int n=0;
  while(n<maxN && fgets(line,sizeof(line),fp)){
    char* p=line;
    a[n]=strtof(p,&p);
    b[n]=strtof(p,&p);
    c[n]=strtof(p,&p);
    n++;
  }
  fclose(fp);
  outN=n;
  return true;
}

static void write_bin_c1(const std::string& fn, const float* a, size_t n){
  std::ofstream ofs(fn, std::ios::binary);
  if(!ofs.is_open()){
    std::cerr<<"ERROR open "<<fn<<"\n";
    std::exit(1);
  }
  ofs.write(reinterpret_cast<const char*>(a), (std::streamsize)(n*sizeof(float)));
  ofs.close();
}

static void write_txt_c1(const std::string& fn, const float* a, size_t n){
  std::ofstream ofs(fn);
  if(!ofs.is_open()){
    std::cerr<<"ERROR open "<<fn<<"\n";
    std::exit(1);
  }
  static thread_local std::vector<char> buf(8<<20);
  ofs.rdbuf()->pubsetbuf(buf.data(), buf.size());
  for(size_t i=0;i<n;i++) ofs<<a[i]<<"\n";
  ofs.close();
}

static void write_txt_viss2(const std::string& fn, const float2* v, size_t n){
  std::ofstream ofs(fn);
  if(!ofs.is_open()){
    std::cerr<<"ERROR open "<<fn<<"\n";
    std::exit(1);
  }
  static thread_local std::vector<char> buf(8<<20);
  ofs.rdbuf()->pubsetbuf(buf.data(), buf.size());
  ofs<<"0 0\n";
  for(size_t i=0;i<n;i++) ofs<<v[i].x<<" "<<v[i].y<<"\n";
  ofs.close();
}

static void phase_correct_viss_host(float2* Viss, const float* w, int n){
  for(int i=0;i<n;i++){
    float ang = -2.0f*(float)M_PI*w[i];
    float c = std::cos(ang);
    float s = std::sin(ang);
    float a = Viss[i].x, b = Viss[i].y;
    Viss[i] = make_float2(a*c - b*s, a*s + b*c);
  }
}

static inline int half_to_full_pos_idx(int ih){
  int group = ih / UNIQUE_BASELINES_PER_T;
  int j = ih - group * UNIQUE_BASELINES_PER_T;
  return group * SIGNED_BASELINES_PER_T + j;
}
static inline int half_to_full_neg_idx(int ih){
  return half_to_full_pos_idx(ih) + UNIQUE_BASELINES_PER_T;
}

static inline bool nearly_neg_pair(float a, float b, float atol=1e-5f, float rtol=1e-5f){
  float ref = fmaxf(fabsf(a), fabsf(b));
  float tol = atol + rtol * ref;
  return fabsf(a + b) <= tol;
}

static bool validate_uvw_halfsym_layout(const std::vector<float>& u,
                                        const std::vector<float>& v,
                                        const std::vector<float>& w,
                                        int N)
{
  if(N % SIGNED_BASELINES_PER_T != 0) return false;
  int T = N / SIGNED_BASELINES_PER_T;
  for(int t=0; t<T; ++t){
    int base = t * SIGNED_BASELINES_PER_T;
    for(int j=0; j<UNIQUE_BASELINES_PER_T; ++j){
      int i0 = base + j;
      int i1 = base + UNIQUE_BASELINES_PER_T + j;
      if(!nearly_neg_pair(u[i0], u[i1]) ||
         !nearly_neg_pair(v[i0], v[i1]) ||
         !nearly_neg_pair(w[i0], w[i1])){
        std::cerr << "ERROR: uvw half-symmetry check failed at group=" << t
                  << " local=" << j
                  << " u=(" << u[i0] << "," << u[i1] << ")"
                  << " v=(" << v[i0] << "," << v[i1] << ")"
                  << " w=(" << w[i0] << "," << w[i1] << ")\n";
        return false;
      }
    }
  }
  return true;
}

static void phase_correct_viss_halfsym_host(float2* Viss_half, const float* w_full, int N_half){
  for(int ih=0; ih<N_half; ++ih){
    int i = half_to_full_pos_idx(ih);
    float ang = -2.0f*(float)M_PI*w_full[i];
    float c = std::cos(ang);
    float s = std::sin(ang);
    float a = Viss_half[ih].x, b = Viss_half[ih].y;
    Viss_half[ih] = make_float2(a*c - b*s, a*s + b*c);
  }
}

static void expand_viss_halfsym_to_full(const std::vector<float2>& hViss_half,
                                        std::vector<float2>& hViss_full)
{
  int N_half = (int)hViss_half.size();
  for(int ih=0; ih<N_half; ++ih){
    int ip = half_to_full_pos_idx(ih);
    int in = half_to_full_neg_idx(ih);
    float2 z = hViss_half[ih];
    hViss_full[ip] = z;
    hViss_full[in] = make_float2(z.x, -z.y);
  }
}

struct OrbitSegSlot {
  float *d_pos=nullptr;
  float *d_u=nullptr,*d_v=nullptr,*d_w=nullptr;
  float *d_x1=nullptr,*d_y1=nullptr,*d_z1=nullptr;
  float *d_x2=nullptr,*d_y2=nullptr,*d_z2=nullptr;

  float *h_pos=nullptr;
  float *h_u=nullptr,*h_v=nullptr,*h_w=nullptr;
  float *h_x1=nullptr,*h_y1=nullptr,*h_z1=nullptr;
  float *h_x2=nullptr,*h_y2=nullptr,*h_z2=nullptr;

  cudaEvent_t ready = nullptr;
  int t0 = 0;
  int tlen = 0;
  int segN = 0;
};

struct OrbitGenCtx {
  int dev = 0;
  cudaStream_t stream = nullptr;
  float lambda_m = 0.0f;
  int dayid = 1;
  uint64_t seed = 12345;

  int OrbitRes = 0;
  int ProcessionCount = 0;
  int segLen = 0;
  int posnum = 0;
  int max_tlen = 0;
  int max_segN = 0;

  float *d_r1=nullptr, *d_r2=nullptr;
  float *d_x=nullptr, *d_y=nullptr, *d_z=nullptr;
  float *d_dx=nullptr, *d_dy=nullptr, *d_dz=nullptr, *d_missv=nullptr;
  int   *d_t1s=nullptr;

  OrbitSegSlot slots[2];
};

void orbit_gen_init(OrbitGenCtx& gen, int dev, float lambda_m, int dayid, int max_tlen, uint64_t seed);
void orbit_gen_make_segment_async(OrbitGenCtx& gen, int slot_idx, int t0, int tlen, bool copy_all_host);
void orbit_gen_wait_slot(OrbitGenCtx& gen, int slot_idx);
void orbit_gen_destroy(OrbitGenCtx& gen);
