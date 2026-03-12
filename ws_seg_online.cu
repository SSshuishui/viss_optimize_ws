// wstack_pipeline_segmented_visswhole.cu
// End-to-end validation pipeline for the current workflow:
//   1) Load full-day uvw/xyza/xyzb (3 columns, first line skipped)
//   2) Compute full-day Viss on GPU WITHOUT baseline batching (one shot), by multi-GPU pixel-chunk split
//      - Always enable geometric blockage in Viss generation
//      - Always apply phase recovery Viss *= exp(-i*2*pi*w)
//   3) Optionally write the phase-corrected Viss to local file
//   4) Segmented (short-time) w-snapshots-style reconstruction on uv-grid
//        - Segment by time instants: N must be divisible by 56 (satnum=8 => 28 pairs => 56 signed baselines)
//        - Each segment builds locg/unique on device0 (CUB) then broadcasts group artifacts to all GPUs
//        - blockage=0: group-average once (fast)
//        - blockage=1: per-pixel visibility mask using xyz, with occ_mode {0,1,2,4} (slow but correct-ish)
//   5) Accumulate segment images -> daily image, then normalize (scalar or per-pixel weight)
//
// IMPORTANT:
// - Viss generation here is direct summation over sky pixels: O(N_baselines * N_pix).
//   This is correct but very heavy for very large nside; use it for correctness testing.
// - This version removes baseline "batch" splitting as requested.
//
// Build:
//  nvcc -O3 --use_fast_math -lineinfo -Xcompiler -fopenmp -std=c++17 \
//    wstack_pipeline_segmented_visswhole.cu -o wstack_pipeline_visswhole
//
// Run example (10M, 2 GPUs):
//   ./wstack_pipeline_visswhole --btag=10M --nside=4096 --day=1 --segs=10 \
//      --blockage=1 --occ_mode=2 \
//      --in_dir=./earth_10Mhz_cuda --sky_dir=./earth_10Mhz --out_dir=./out10M_fullpipe_v2/ \
//      --gpus=1,2 --uvw_max=4500000 --B_clip=500000 --write_viss=1
//
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
  cudaError_t e=(call); \
  if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

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
  std::string cmd="mkdir -p "+path;
  (void)std::system(cmd.c_str());
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

  const std::string b1 = sky_dir + "/B_" + btag + ".bin";
  const std::string b2 = sky_dir + "/B.bin";
  const std::string t1 = sky_dir + "/B_" + btag + ".txt";
  const std::string t2 = sky_dir + "/B.txt";

  if (mode == "bin") return try_bin(b1) || try_bin(b2);
  if (mode == "txt") return try_txt(t1) || try_txt(t2);
  return try_bin(b1) || try_bin(b2) || try_txt(t1) || try_txt(t2);
}

static void write_txt_3cols_stream(std::ofstream& ofs,
                                   const float* a, const float* b, const float* c,
                                   size_t n) {
  for (size_t i=0; i<n; ++i) ofs << a[i] << " " << b[i] << " " << c[i] << "\n";
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

__device__ __forceinline__ void sincos_fast(float x,float* s,float* c){ __sincosf(x,s,c); }
__device__ __forceinline__ float2 cadd(float2 a,float2 b){ return make_float2(a.x+b.x,a.y+b.y); }
__device__ __forceinline__ float2 cmul(float2 a,float2 b){ return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ __forceinline__ float2 cexpj(float phase){ float s,c; __sincosf(phase,&s,&c); return make_float2(c,s); }
__device__ __forceinline__ int round_away_from_zero(float x){ return (x>=0.0f)? (int)floorf(x+0.5f) : (int)ceilf(x-0.5f); }


// -------- NEST pixel -> l,m,n directly on GPU --------
__device__ __forceinline__ float clamp01(float x) { return fminf(1.0f, fmaxf(-1.0f, x)); }
__device__ __forceinline__ void deinterleave_10(unsigned int ip, int &x, int &y) {
  x = 0; y = 0;
  #pragma unroll
  for (int b = 0; b < 5; ++b) {
    x |= ((ip >> (2*b))     & 1u) << b;
    y |= ((ip >> (2*b + 1)) & 1u) << b;
  }
}
__constant__ int c_jrll[12] = {2,2,2,2,3,3,3,3,4,4,4,4};
__constant__ int c_jpll[12] = {1,3,5,7,0,2,4,6,1,3,5,7};

__global__ void pix2lmn_nest_kernel(
    int nside,
    unsigned int base_ipix,
    int chunkN,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;

  unsigned int ipix = base_ipix + (unsigned int)tid;
  unsigned int npface = (unsigned int)nside * (unsigned int)nside;
  unsigned int face_num = ipix / npface;
  unsigned int ipf      = ipix - face_num*npface;

  int ix = 0, iy = 0;
  unsigned int v = ipf;
  int scalemlv = 1;

  #pragma unroll
  for (int k = 0; k < 5; ++k) {
    unsigned int low = v & 1023u;
    int x, y;
    deinterleave_10(low, x, y);
    ix += scalemlv * x;
    iy += scalemlv * y;
    scalemlv <<= 5;
    v >>= 10;
  }
  {
    unsigned int low = v & 1023u;
    int x, y;
    deinterleave_10(low, x, y);
    ix += scalemlv * x;
    iy += scalemlv * y;
  }

  int jrt = ix + iy;
  int jpt = ix - iy;
  int nl4 = 4 * nside;
  int jr  = c_jrll[face_num] * nside - jrt - 1;

  float fact1 = 1.0f / (3.0f * (float)nside * (float)nside);
  float fact2 = 2.0f / (3.0f * (float)nside);

  int nr, kshift;
    float z;
  if (jr < nside) {
    nr = jr;
    z = 1.0f - (float)(nr * nr) * fact1;
    kshift = 0;
  } else if (jr <= 3*nside) {
    nr = nside;
    z = (float)(2*nside - jr) * fact2;
    kshift = (jr - nside) & 1;
  } else {
    nr = nl4 - jr;
    z = -1.0f + (float)(nr * nr) * fact1;
    kshift = 0;
  }

  z = clamp01(z);
  float theta = acosf(z);

  int jp = ((c_jpll[face_num]*nr) + jpt + 1 + kshift) >> 1;
  if (jp > nl4) jp -= nl4;
  if (jp < 1)   jp += nl4;

  float phi = (0.5f * (float)M_PI) * (((float)jp) - 0.5f*(float)(kshift + 1)) / (float)nr;
  if (phi < 0.0f) phi += 2.0f * (float)M_PI;

  float th = (float)M_PI * 0.5f - theta;
  if(phi > (float)M_PI) phi -= 2.0f*(float)M_PI;
  phi = -phi;

  float st, ct, sp, cp;
  sincos_fast(th, &st, &ct);
  sincos_fast(phi, &sp, &cp);

  l[tid] = ct * cp;
  m[tid] = ct * sp;
  n[tid] = st;
}

// -------- online orbit / uvw generator --------
static constexpr int   SATNUM = 8;
static constexpr float C0     = 3e8f;
static constexpr float A_KM   = 2038.14f;
static constexpr float INCL   = 30.0f * (float)(M_PI / 180.0);
static constexpr float ARGP   = 0.0f;
static constexpr float ORBIT_HOURS = 2.3f;
static constexpr float RAAN_STEP_DEG_PER_REV = 0.08f;
static constexpr float RAAN_STEP_DEG_PER_K   = 0.08f;
static constexpr float RAAN_DAY_DEG          = 0.8f;
static constexpr int   MAX_AMOUNT = 64;

__constant__ int d_pair_m[MAX_AMOUNT];
__constant__ int d_pair_n[MAX_AMOUNT];

static inline int round_away_from_zero_host(float x) {
  return (x >= 0.0f) ? (int)floorf(x + 0.5f) : (int)ceilf(x - 0.5f);
}
static inline int mod_matlab_int_host(int a, int m) {
  int r = a % m;
  return (r < 0) ? (r + m) : r;
}
static void init_pair_constants() {
  int hm[MAX_AMOUNT], hn[MAX_AMOUNT];
  int amount = 0;
  for (int m=0; m<SATNUM-1; ++m) {
    for (int n=m+1; n<SATNUM; ++n) {
      hm[amount] = m;
      hn[amount] = n;
      amount++;
    }
  }
  CHECK_CUDA(cudaMemcpyToSymbol(d_pair_m, hm, amount*sizeof(int)));
  CHECK_CUDA(cudaMemcpyToSymbol(d_pair_n, hn, amount*sizeof(int)));
}

__device__ __forceinline__
float Mt_kf(int k0, int OrbitRes) {
  const float start = 1.0f / (float)OrbitRes;
  const float end   = (float)(2.0 * M_PI + M_PI / 200.0);
  if (OrbitRes <= 1) return start;
  const float t = (float)k0 / (float)(OrbitRes - 1);
  return start + (end - start) * t;
}

__global__
void k_compute_sat1_xyz(float* x, float* y, float* z,
                        int OrbitRes, int ProcessionCount, int dayid)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int Nrows = OrbitRes * ProcessionCount;
  if (idx >= Nrows) return;

  const int g = idx / OrbitRes;
  const int k = idx - g * OrbitRes;

  const float d2r = (float)(M_PI / 180.0);
  const float raan =
      (float)g * d2r * RAAN_STEP_DEG_PER_REV
    + ((float)(k+1) / (float)OrbitRes) * d2r * RAAN_STEP_DEG_PER_K
    + (float)(dayid - 1) * d2r * RAAN_DAY_DEG;

  const float M = Mt_kf(k, OrbitRes);
  const float theta = M;
  const float r = A_KM;
  const float ww = theta + ARGP;

  float sw, cw, sr, cr, si, ci;
  sincosf(ww, &sw, &cw);
  sincosf(raan, &sr, &cr);
  sincosf(INCL, &si, &ci);

  const float X = r * (cw*cr - sw*ci*sr);
  const float Y = r * (cw*sr + sw*ci*cr);
  const float Z = r * (sw*si);

  const int base = idx * SATNUM;
  x[base + 0] = X;
  y[base + 0] = Y;
  z[base + 0] = Z;
}

__global__
void k_diff_sat1(const float* x, const float* y, const float* z,
                 float* dx, float* dy, float* dz,
                 int OrbitRes, int ProcessionCount)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int Nrows = OrbitRes * ProcessionCount;
  const int Nv = Nrows - 1;
  if (i >= Nv) return;

  const int a0 = i * SATNUM;
  const int a1 = (i+1) * SATNUM;
  dx[i] = x[a1+0] - x[a0+0];
  dy[i] = y[a1+0] - y[a0+0];
  dz[i] = z[a1+0] - z[a0+0];
}

__global__
void k_norm3(const float* dx, const float* dy, const float* dz, float* missv, int n)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float a = dx[i], b = dy[i], c = dz[i];
  missv[i] = sqrtf(a*a + b*b + c*c);
}

__device__ __forceinline__
float triangle_linspace_f(float r2, float r1, int p, int T)
{
  if (T <= 1) return r2;
  if (p < T) {
    float t = (float)p / (float)(T - 1);
    return r2 + (r1 - r2) * t;
  } else {
    int q = p - T;
    float t = (float)q / (float)(T - 1);
    return r1 + (r2 - r1) * t;
  }
}

__global__
void k_compute_other_sats(float* x, float* y, float* z,
                          const float* dx, const float* dy, const float* dz,
                          const float* missv,
                          const float* r1, const float* r2,
                          int OrbitRes, int ProcessionCount,
                          int T, int day_offset)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int Nrows = OrbitRes * ProcessionCount;
  const int Nv = Nrows - 1;
  if (i >= Nv) return;

  const int period = 2 * T;
  const int p = (period > 0) ? ((day_offset + i) % period) : 0;

  const float vx = dx[i], vy = dy[i], vz = dz[i];
  const float mv = missv[i] + 1e-20f;

  const int row = i + 1;
  const int base_ref = row * SATNUM;

  const float xr = x[base_ref + 0];
  const float yr = y[base_ref + 0];
  const float zr = z[base_ref + 0];

  #pragma unroll
  for (int sat = 1; sat < SATNUM; ++sat) {
    const float rt = triangle_linspace_f(r2[sat], r1[sat], p, T);
    const float ddx = rt * vx / mv / 1000.0f;
    const float ddy = rt * vy / mv / 1000.0f;
    const float ddz = rt * vz / mv / 1000.0f;
    x[base_ref + sat] = xr - ddx;
    y[base_ref + sat] = yr - ddy;
    z[base_ref + sat] = zr - ddz;
  }
}

__global__
void k_gather_pos_range(const float* x, const float* y, const float* z,
                        float* pos,
                        const int* t1s,
                        int OrbitRes, int ProcessionCount, int segLen,
                        int base_out_row, int out_rows)
{
  const int outRowLocal = blockIdx.x * blockDim.x + threadIdx.x;
  if (outRowLocal >= out_rows) return;

  const int outRowGlobal = base_out_row + outRowLocal;
  const int orb = outRowGlobal / segLen;
  const int loc = outRowGlobal - orb * segLen;
  const int srcRow = t1s[orb] + loc;

  const int srcBase = srcRow * SATNUM;
  const int dstBase = outRowLocal * (3 * SATNUM);

  for (int s=0; s<SATNUM; ++s) pos[dstBase + s] = x[srcBase + s];
  for (int s=0; s<SATNUM; ++s) pos[dstBase + SATNUM + s] = y[srcBase + s];
  for (int s=0; s<SATNUM; ++s) pos[dstBase + 2*SATNUM + s] = z[srcBase + s];
}

__global__
void k_compute_uvw_xyz(const float* pos, int posnum, float lambda_m,
                       float* u, float* v, float* w,
                       float* x1, float* y1, float* z1,
                       float* x2, float* y2, float* z2)
{
  const int amount = SATNUM * (SATNUM - 1) / 2;
  const int outN = 2 * amount * posnum;
  const int outIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (outIdx >= outN) return;

  const int basePerTime = 2 * amount;
  const int timeIdx = outIdx / basePerTime;
  const int baseIdx = outIdx - timeIdx * basePerTime;

  const int k = baseIdx % amount;
  const int m = d_pair_m[k];
  const int n = d_pair_n[k];
  const float sgn = (baseIdx < amount) ? -1.0f : +1.0f;
    const int base = timeIdx * (3 * SATNUM);

  const float xm = pos[base + m];
  const float ym = pos[base + SATNUM + m];
  const float zm = pos[base + 2*SATNUM + m];

  const float xn = pos[base + n];
  const float yn = pos[base + SATNUM + n];
  const float zn = pos[base + 2*SATNUM + n];

  const float du = sgn * (xn - xm) * 1e3f / lambda_m;
  const float dv = sgn * (yn - ym) * 1e3f / lambda_m;
  const float dw = sgn * (zn - zm) * 1e3f / lambda_m;

  u[outIdx] = du;
  v[outIdx] = dv;
  w[outIdx] = dw;

  if (baseIdx < amount) {
    x1[outIdx] = xn; y1[outIdx] = yn; z1[outIdx] = zn;
    x2[outIdx] = xm; y2[outIdx] = ym; z2[outIdx] = zm;
  } else {
    x1[outIdx] = xm; y1[outIdx] = ym; z1[outIdx] = zm;
    x2[outIdx] = xn; y2[outIdx] = yn; z2[outIdx] = zn;
  }
}

struct OrbitGenCtx {
  int dev = 0;
  cudaStream_t stream = nullptr;
  float lambda_m = 0.0f;
  int dayid = 1;
  uint64_t seed = 12345;

  int OrbitRes = 0;
  int ProcessionCount = 0;
  int segLen = 0;
  int posnum = 0;      // total T
  int max_tlen = 0;
  int max_segN = 0;

  float *d_r1=nullptr, *d_r2=nullptr;
  float *d_x=nullptr, *d_y=nullptr, *d_z=nullptr;
  float *d_dx=nullptr, *d_dy=nullptr, *d_dz=nullptr, *d_missv=nullptr;
  int   *d_t1s=nullptr;
  float *d_pos=nullptr;
  float *d_u=nullptr,*d_v=nullptr,*d_w=nullptr;
  float *d_x1=nullptr,*d_y1=nullptr,*d_z1=nullptr;
  float *d_x2=nullptr,*d_y2=nullptr,*d_z2=nullptr;

  float *h_u=nullptr,*h_v=nullptr,*h_w=nullptr;
  float *h_x1=nullptr,*h_y1=nullptr,*h_z1=nullptr;
  float *h_x2=nullptr,*h_y2=nullptr,*h_z2=nullptr;
};

static void orbit_gen_init(OrbitGenCtx& gen, int dev, float lambda_m, int dayid, int max_tlen, uint64_t seed){
  gen.dev = dev;
  gen.lambda_m = lambda_m;
  gen.dayid = dayid;
  gen.seed = seed;
  gen.OrbitRes = (int)std::ceil((double)(2.0 * M_PI) * (double)(100e3 / (double)lambda_m));
  gen.ProcessionCount = round_away_from_zero_host(24.0f / ORBIT_HOURS);
  gen.segLen = gen.OrbitRes / 3;
  gen.posnum = gen.ProcessionCount * gen.segLen;
  gen.max_tlen = max_tlen;
  gen.max_segN = max_tlen * SIGNED_BASELINES_PER_T;

  CHECK_CUDA(cudaSetDevice(dev));
  CHECK_CUDA(cudaStreamCreateWithFlags(&gen.stream, cudaStreamNonBlocking));
  init_pair_constants();

  float h_r1[SATNUM] = {0, 100e3f/23.0f, 4*(100e3f/23.0f), 10*(100e3f/23.0f), 16*(100e3f/23.0f), 18*(100e3f/23.0f), 21*(100e3f/23.0f), 23*(100e3f/23.0f)};
  float h_r2[SATNUM] = {0, 0.1e3f, 0.4f*(100e3f/23.0f), 1*(100e3f/23.0f), 1.6f*(100e3f/23.0f), 1.8f*(100e3f/23.0f), 2.1f*(100e3f/23.0f), 2.3f*(100e3f/23.0f)};

  const int Nrows = gen.OrbitRes * gen.ProcessionCount;
  const int Nv = Nrows - 1;

  CHECK_CUDA(cudaMalloc(&gen.d_r1, SATNUM*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_r2, SATNUM*sizeof(float)));
  CHECK_CUDA(cudaMemcpyAsync(gen.d_r1, h_r1, SATNUM*sizeof(float), cudaMemcpyHostToDevice, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(gen.d_r2, h_r2, SATNUM*sizeof(float), cudaMemcpyHostToDevice, gen.stream));

  CHECK_CUDA(cudaMalloc(&gen.d_x, (size_t)Nrows * SATNUM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_y, (size_t)Nrows * SATNUM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_z, (size_t)Nrows * SATNUM * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_dx, (size_t)Nv * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_dy, (size_t)Nv * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_dz, (size_t)Nv * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_missv, (size_t)Nv * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_t1s, (size_t)gen.ProcessionCount * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&gen.d_pos, (size_t)max_tlen * (3*SATNUM) * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_u,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_v,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_w,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_x1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_y1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_z1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_x2, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_y2, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&gen.d_z2, (size_t)gen.max_segN * sizeof(float)));

  CHECK_CUDA(cudaMallocHost(&gen.h_u,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_v,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_w,  (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_x1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_y1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_z1, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_x2, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_y2, (size_t)gen.max_segN * sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&gen.h_z2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMemsetAsync(gen.d_x, 0, (size_t)Nrows * SATNUM * sizeof(float), gen.stream));
  CHECK_CUDA(cudaMemsetAsync(gen.d_y, 0, (size_t)Nrows * SATNUM * sizeof(float), gen.stream));
  CHECK_CUDA(cudaMemsetAsync(gen.d_z, 0, (size_t)Nrows * SATNUM * sizeof(float), gen.stream));

  const int BS = 256;
  int GS = (Nrows + BS - 1) / BS;
  k_compute_sat1_xyz<<<GS, BS, 0, gen.stream>>>(gen.d_x, gen.d_y, gen.d_z, gen.OrbitRes, gen.ProcessionCount, dayid);
  CHECK_CUDA(cudaPeekAtLastError());

  GS = (Nv + BS - 1) / BS;
  k_diff_sat1<<<GS, BS, 0, gen.stream>>>(gen.d_x, gen.d_y, gen.d_z, gen.d_dx, gen.d_dy, gen.d_dz, gen.OrbitRes, gen.ProcessionCount);
  CHECK_CUDA(cudaPeekAtLastError());
  k_norm3<<<GS, BS, 0, gen.stream>>>(gen.d_dx, gen.d_dy, gen.d_dz, gen.d_missv, Nv);
  CHECK_CUDA(cudaPeekAtLastError());

  {
    const int T = round_away_from_zero_host((float)gen.OrbitRes * 7.0f * 24.0f / ORBIT_HOURS);
    const int mod14 = mod_matlab_int_host(dayid - 1, 14);
    const int day_offset = round_away_from_zero_host((float)mod14 * (float)gen.OrbitRes * 24.0f / ORBIT_HOURS);
    k_compute_other_sats<<<GS, BS, 0, gen.stream>>>(gen.d_x, gen.d_y, gen.d_z,
                                                    gen.d_dx, gen.d_dy, gen.d_dz, gen.d_missv,
                                                    gen.d_r1, gen.d_r2,
                                                    gen.OrbitRes, gen.ProcessionCount,
                                                    T, day_offset);
    CHECK_CUDA(cudaPeekAtLastError());
  }

  {
    uint64_t day_seed = seed ^ (uint64_t)dayid * 0x9E3779B97F4A7C15ULL;
    std::mt19937_64 rng(day_seed);
    std::vector<int> h_t1s(gen.ProcessionCount);
    for (int g=0; g<gen.ProcessionCount; ++g) {
      const int orbitStart = g * gen.OrbitRes;
      const int orbitEnd   = (g+1)*gen.OrbitRes - 1;
      double u01 = (double)(rng() >> 11) * (1.0 / 9007199254740992.0);
      int t1 = orbitStart + round_away_from_zero_host((float)(u01 * (double)gen.OrbitRes * (2.0/3.0)));
      int t2 = t1 + gen.segLen - 1;
      if (t2 > orbitEnd) { t2 = orbitEnd; t1 = t2 - gen.segLen + 1; }
      if (t1 < orbitStart) t1 = orbitStart;
      if (t1 + gen.segLen - 1 > orbitEnd) t1 = orbitEnd - gen.segLen + 1;
      h_t1s[g] = t1;
    }
    CHECK_CUDA(cudaMemcpyAsync(gen.d_t1s, h_t1s.data(), (size_t)gen.ProcessionCount*sizeof(int), cudaMemcpyHostToDevice, gen.stream));
  }

  CHECK_CUDA(cudaStreamSynchronize(gen.stream));
}

static void orbit_gen_make_segment(OrbitGenCtx& gen, int t0, int tlen, bool copy_all_host){
  CHECK_CUDA(cudaSetDevice(gen.dev));
  const int BS = 256;
  int GS = (tlen + BS - 1) / BS;
  k_gather_pos_range<<<GS, BS, 0, gen.stream>>>(gen.d_x, gen.d_y, gen.d_z, gen.d_pos, gen.d_t1s,
                                                gen.OrbitRes, gen.ProcessionCount, gen.segLen, t0, tlen);
  CHECK_CUDA(cudaPeekAtLastError());

  int segN = tlen * SIGNED_BASELINES_PER_T;
  GS = (segN + BS - 1) / BS;
  k_compute_uvw_xyz<<<GS, BS, 0, gen.stream>>>(gen.d_pos, tlen, gen.lambda_m,
                                               gen.d_u, gen.d_v, gen.d_w,
                                               gen.d_x1, gen.d_y1, gen.d_z1,
                                               gen.d_x2, gen.d_y2, gen.d_z2);
  CHECK_CUDA(cudaPeekAtLastError());

  CHECK_CUDA(cudaMemcpyAsync(gen.h_u, gen.d_u, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(gen.h_v, gen.d_v, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(gen.h_w, gen.d_w, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  if(copy_all_host){
    CHECK_CUDA(cudaMemcpyAsync(gen.h_x1, gen.d_x1, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(gen.h_y1, gen.d_y1, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(gen.h_z1, gen.d_z1, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(gen.h_x2, gen.d_x2, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(gen.h_y2, gen.d_y2, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(gen.h_z2, gen.d_z2, (size_t)segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(gen.stream));
}

static void orbit_gen_destroy(OrbitGenCtx& gen){
  CHECK_CUDA(cudaSetDevice(gen.dev));
  if(gen.h_z2) cudaFreeHost(gen.h_z2);
  if(gen.h_y2) cudaFreeHost(gen.h_y2);
  if(gen.h_x2) cudaFreeHost(gen.h_x2);
  if(gen.h_z1) cudaFreeHost(gen.h_z1);
  if(gen.h_y1) cudaFreeHost(gen.h_y1);
  if(gen.h_x1) cudaFreeHost(gen.h_x1);
  if(gen.h_w) cudaFreeHost(gen.h_w);
  if(gen.h_v) cudaFreeHost(gen.h_v);
  if(gen.h_u) cudaFreeHost(gen.h_u);

  if(gen.d_z2) cudaFree(gen.d_z2);
  if(gen.d_y2) cudaFree(gen.d_y2);
  if(gen.d_x2) cudaFree(gen.d_x2);
  if(gen.d_z1) cudaFree(gen.d_z1);
  if(gen.d_y1) cudaFree(gen.d_y1);
  if(gen.d_x1) cudaFree(gen.d_x1);
  if(gen.d_w) cudaFree(gen.d_w);
  if(gen.d_v) cudaFree(gen.d_v);
  if(gen.d_u) cudaFree(gen.d_u);
  if(gen.d_pos) cudaFree(gen.d_pos);
  if(gen.d_t1s) cudaFree(gen.d_t1s);
  if(gen.d_missv) cudaFree(gen.d_missv);
  if(gen.d_dz) cudaFree(gen.d_dz);
  if(gen.d_dy) cudaFree(gen.d_dy);
  if(gen.d_dx) cudaFree(gen.d_dx);
  if(gen.d_z) cudaFree(gen.d_z);
  if(gen.d_y) cudaFree(gen.d_y);
    if(gen.d_x) cudaFree(gen.d_x);
  if(gen.d_r2) cudaFree(gen.d_r2);
  if(gen.d_r1) cudaFree(gen.d_r1);
  if(gen.stream) cudaStreamDestroy(gen.stream);
}

static bool validate_uvw_halfsym_layout_ptr(const float* u, const float* v, const float* w, int N)
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
         !nearly_neg_pair(w[i0], w[i1])) return false;
    }
  }
  return true;
}

__global__ void healpix_lmn_from_theta_phi_chunk(
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    long long n_chunk)
{
  long long idx=(long long)blockIdx.x*blockDim.x + threadIdx.x;
  if(idx>=n_chunk) return;
  float th=theta[idx];
  float ph=phi[idx];
  th=(float)M_PI*0.5f - th;
  if(ph>(float)M_PI) ph-=2.0f*(float)M_PI;
  ph=-ph;
  float st,ct,sp,cp;
  sincos_fast(th,&st,&ct);
  sincos_fast(ph,&sp,&cp);
  l[idx]=ct*cp;
  m[idx]=ct*sp;
  n[idx]=st;
}

__global__ void invnorm3_kernel(const float* x,const float* y,const float* z,float* invn,int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    float a=x[i], b=y[i], c=z[i];
    invn[i]=rsqrtf(a*a+b*b+c*c);
  }
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (i < N);

  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;

  if(active){
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for(long long p0 = 0; p0 < n_chunk; p0 += TILE_PIX){
    for(int lane = threadIdx.x; lane < TILE_PIX; lane += blockDim.x){
      long long p = p0 + lane;
      if(p < n_chunk){
        sB[lane] = B[p];
        sL[lane] = l[p];
        sM[lane] = m[p];
        sN[lane] = n[p];
      }else{
        sB[lane] = 0.0f; sL[lane] = 0.0f; sM[lane] = 0.0f; sN[lane] = 0.0f;
      }
    }
    __syncthreads();

    if(active){
      int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
      #pragma unroll 4
      for(int k0=0; k0<tileN; ++k0){
        float lp = sL[k0], mp = sM[k0], npv = sN[k0];
                if constexpr (DO_BLOCKAGE){
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
        }
        float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
        float ang = k*phase;
        float s,c; sincos_fast(ang,&s,&c);
        float bp=sB[k0];
        acc_re += bp*c;
        acc_im += bp*s;
      }
    }
    __syncthreads();
  }

  if(active) Vpart[i] = make_float2(acc_re, acc_im);
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  int i = 0;
  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;

  if(active){
    int group = ih / UNIQUE_BASELINES_PER_T;
    int j     = ih - group * UNIQUE_BASELINES_PER_T;
    i         = group * SIGNED_BASELINES_PER_T + j;
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for(long long p0 = 0; p0 < n_chunk; p0 += TILE_PIX){
    for(int lane = threadIdx.x; lane < TILE_PIX; lane += blockDim.x){
      long long p = p0 + lane;
      if(p < n_chunk){
        sB[lane] = B[p];
        sL[lane] = l[p];
        sM[lane] = m[p];
        sN[lane] = n[p];
      }else{
        sB[lane] = 0.0f; sL[lane] = 0.0f; sM[lane] = 0.0f; sN[lane] = 0.0f;
      }
    }
    __syncthreads();

    if(active){
      int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
      #pragma unroll 4
      for(int k0=0; k0<tileN; ++k0){
        float lp = sL[k0], mp = sM[k0], npv = sN[k0];
        if constexpr (DO_BLOCKAGE){
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
        }
        float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
        float ang = k*phase;
        float s,c; sincos_fast(ang,&s,&c);
        float bp=sB[k0];
        acc_re += bp*c;
        acc_im += bp*s;
      }
    }
    __syncthreads();
  }

  if(active) Vpart[ih] = make_float2(acc_re, acc_im);
}

// ----- grouping (device0) -----
__global__ void iota_kernel(int* idx, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) idx[i]=i;
}

__global__ void build_locg_kernel(const float* __restrict__ u,
                                  const float* __restrict__ v,
                                  int* __restrict__ locg,
                                  int n,
                                  float inv_du,
                                  int RES,
                                  int half)
{
            if constexpr (DO_BLOCKAGE){
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
        }
        float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
        float ang = k*phase;
        float s,c; sincos_fast(ang,&s,&c);
        float bp=sB[k0];
        acc_re += bp*c;
        acc_im += bp*s;
      }
    }
    __syncthreads();
  }

  if(active) Vpart[i] = make_float2(acc_re, acc_im);
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  int i = 0;
  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;

  if(active){
    int group = ih / UNIQUE_BASELINES_PER_T;
    int j     = ih - group * UNIQUE_BASELINES_PER_T;
    i         = group * SIGNED_BASELINES_PER_T + j;
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for(long long p0 = 0; p0 < n_chunk; p0 += TILE_PIX){
    for(int lane = threadIdx.x; lane < TILE_PIX; lane += blockDim.x){
      long long p = p0 + lane;
      if(p < n_chunk){
        sB[lane] = B[p];
        sL[lane] = l[p];
        sM[lane] = m[p];
        sN[lane] = n[p];
      }else{
        sB[lane] = 0.0f; sL[lane] = 0.0f; sM[lane] = 0.0f; sN[lane] = 0.0f;
      }
    }
    __syncthreads();

    if(active){
      int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
      #pragma unroll 4
      for(int k0=0; k0<tileN; ++k0){
        float lp = sL[k0], mp = sM[k0], npv = sN[k0];
        if constexpr (DO_BLOCKAGE){
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
        }
        float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
        float ang = k*phase;
        float s,c; sincos_fast(ang,&s,&c);
        float bp=sB[k0];
        acc_re += bp*c;
        acc_im += bp*s;
      }
    }
    __syncthreads();
  }

  if(active) Vpart[ih] = make_float2(acc_re, acc_im);
}

// ----- grouping (device0) -----
__global__ void iota_kernel(int* idx, int n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) idx[i]=i;
}

__global__ void build_locg_kernel(const float* __restrict__ u,
                                  const float* __restrict__ v,
                                  int* __restrict__ locg,
                                  int n,
                                  float inv_du,
                                  int RES,
                                  int half)
{
      int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    int ui=round_away_from_zero(u[i]*inv_du);
    int vi=round_away_from_zero(v[i]*inv_du);
    int U=ui+half;
    int V=vi+half;
    if((unsigned)U>=(unsigned)RES || (unsigned)V>=(unsigned)RES) locg[i]=0;
    else locg[i]=U*RES + V + 1;
  }
}

__global__ void set_last_offset(int* offsets, int nuniq, int n){
  if(blockIdx.x==0 && threadIdx.x==0) offsets[nuniq]=n;
}

__global__ void gather_viss_by_idx(const float2* __restrict__ viss_in,
                                   const int* __restrict__ idx_sorted,
                                   float2* __restrict__ viss_sorted,
                                   int n)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) viss_sorted[i]=viss_in[idx_sorted[i]];
}

__global__ void compute_avg(float2* avg, const float2* sum, const int* counts, int nuniq){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<nuniq){
    int c=counts[i];
    avg[i]=(c>0)? make_float2(sum[i].x/c, sum[i].y/c) : make_float2(0,0);
  }
}

struct Float2AddOp {
  __host__ __device__ float2 operator()(const float2& a,const float2& b) const {
    return make_float2(a.x+b.x,a.y+b.y);
  }
};

struct GroupArtifactsHost {
  int nuniq=0;
  std::vector<int> keys_unique;
  std::vector<float2> viss_avg;    // blockage=0
  std::vector<int> offsets;        // blockage=1
  std::vector<int> idx_sorted;     // blockage=1
};

static void build_groups_device0(
  int dev, cudaStream_t stream,
  const float* d_u_seg, const float* d_v_seg,
  const float2* d_viss_seg,
  int n,
  float du, int RES, int half,
  int blockage,
  GroupArtifactsHost &hout)
{
  CHECK_CUDA(cudaSetDevice(dev));
  float inv_du=1.0f/du;

  int *d_locg=nullptr, *d_idx=nullptr, *d_idx_tmp=nullptr;
  int *d_keys_in=nullptr, *d_keys_tmp=nullptr;
  CHECK_CUDA(cudaMalloc(&d_locg, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx_tmp, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_keys_in, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_keys_tmp, n*sizeof(int)));

  int block=256, grid=(n+block-1)/block;
  build_locg_kernel<<<grid,block,0,stream>>>(d_u_seg,d_v_seg,d_locg,n,inv_du,RES,half);
  iota_kernel<<<grid,block,0,stream>>>(d_idx,n);
  CHECK_CUDA(cudaPeekAtLastError());
  CHECK_CUDA(cudaMemcpyAsync(d_keys_in,d_locg,n*sizeof(int),cudaMemcpyDeviceToDevice,stream));

  cub::DoubleBuffer<int> keys(d_keys_in, d_keys_tmp);
  cub::DoubleBuffer<int> vals(d_idx, d_idx_tmp);

  void* d_temp=nullptr;
  size_t temp_bytes=0;
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,keys,vals,n,0,8*sizeof(int),stream));
  CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,keys,vals,n,0,8*sizeof(int),stream));
  CHECK_CUDA(cudaFree(d_temp));

  int* d_keys_sorted=keys.Current();
  int* d_idx_sorted=vals.Current();

  int *d_keys_unique=nullptr, *d_counts=nullptr, *d_num_runs=nullptr;
  CHECK_CUDA(cudaMalloc(&d_keys_unique, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_counts, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_num_runs, sizeof(int)));

  d_temp=nullptr;
  temp_bytes=0;
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp,temp_bytes,d_keys_sorted,d_keys_unique,d_counts,d_num_runs,n,stream));
  CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp,temp_bytes,d_keys_sorted,d_keys_unique,d_counts,d_num_runs,n,stream));
  CHECK_CUDA(cudaFree(d_temp));

  int nuniq=0;
  CHECK_CUDA(cudaMemcpyAsync(&nuniq,d_num_runs,sizeof(int),cudaMemcpyDeviceToHost,stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaFree(d_num_runs));

  hout.nuniq=nuniq;
  hout.keys_unique.assign(nuniq,0);
  CHECK_CUDA(cudaMemcpy(hout.keys_unique.data(), d_keys_unique, nuniq*sizeof(int), cudaMemcpyDeviceToHost));

  if(blockage==0){
    float2* d_viss_sorted=nullptr;
    CHECK_CUDA(cudaMalloc(&d_viss_sorted, n*sizeof(float2)));
    gather_viss_by_idx<<<grid,block,0,stream>>>(d_viss_seg,d_idx_sorted,d_viss_sorted,n);

    int* d_keys_out2=nullptr;
    float2* d_sum=nullptr;
    int* d_num2=nullptr;
    CHECK_CUDA(cudaMalloc(&d_keys_out2, nuniq*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sum, nuniq*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&d_num2, sizeof(int)));

    d_temp=nullptr;
    temp_bytes=0;
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,temp_bytes,d_keys_sorted,d_keys_out2,d_viss_sorted,d_sum,d_num2,Float2AddOp(),n,stream));
    CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,temp_bytes,d_keys_sorted,d_keys_out2,d_viss_sorted,d_sum,d_num2,Float2AddOp(),n,stream));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_keys_out2));
    CHECK_CUDA(cudaFree(d_num2));
    CHECK_CUDA(cudaFree(d_viss_sorted));

    float2* d_avg=nullptr;
    CHECK_CUDA(cudaMalloc(&d_avg, nuniq*sizeof(float2)));
    int g2=(nuniq+block-1)/block;
    compute_avg<<<g2,block,0,stream>>>(d_avg,d_sum,d_counts,nuniq);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    hout.viss_avg.assign(nuniq, make_float2(0,0));
    CHECK_CUDA(cudaMemcpy(hout.viss_avg.data(), d_avg, nuniq*sizeof(float2), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_avg));
    CHECK_CUDA(cudaFree(d_sum));
  } else {
    int* d_offsets=nullptr;
    CHECK_CUDA(cudaMalloc(&d_offsets,(nuniq+1)*sizeof(int)));
    d_temp=nullptr;
    temp_bytes=0;
    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp,temp_bytes,d_counts,d_offsets,nuniq,stream));
    CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
    CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp,temp_bytes,d_counts,d_offsets,nuniq,stream));
    CHECK_CUDA(cudaFree(d_temp));
    set_last_offset<<<1,1,0,stream>>>(d_offsets,nuniq,n);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    hout.offsets.assign(nuniq+1,0);
    hout.idx_sorted.assign(n,0);
    CHECK_CUDA(cudaMemcpy(hout.offsets.data(), d_offsets, (nuniq+1)*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hout.idx_sorted.data(), d_idx_sorted, n*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_offsets));
  }

  CHECK_CUDA(cudaFree(d_keys_unique));
  CHECK_CUDA(cudaFree(d_counts));
  CHECK_CUDA(cudaFree(d_locg));
  CHECK_CUDA(cudaFree(d_idx));
  CHECK_CUDA(cudaFree(d_idx_tmp));
  CHECK_CUDA(cudaFree(d_keys_in));
  CHECK_CUDA(cudaFree(d_keys_tmp));
}

// ----- recon kernels -----
template<int CHUNK_KEYS>
__global__ void recon_seg_keys_avg(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const float2* __restrict__ viss_avg,
    int nuniq,
    int RES, int half, float du,
    float fa, float fb,
    float* __restrict__ Cseg)
{
  __shared__ int shK[CHUNK_KEYS];
  __shared__ float2 shV[CHUNK_KEYS];
  long long pix=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(pix>=n_chunk) return;

  float lp=l[pix] + fa*n[pix];
  float mp=m[pix] + fb*n[pix];
  float2 acc=make_float2(0,0);
  const float TWO_PI=6.2831853071795864769f;

  for(int base=0;base<nuniq;base+=CHUNK_KEYS){
    int t=threadIdx.x;
    if(t<CHUNK_KEYS){
      int j=base+t;
      if(j<nuniq){ shK[t]=keys_unique[j]; shV[t]=viss_avg[j]; }
      else { shK[t]=0; shV[t]=make_float2(0,0); }
    }
    __syncthreads();
    #pragma unroll
    for(int k=0;k<CHUNK_KEYS;k++){
      int key=shK[k];
      if(key==0) continue;
      int tmp=key-1;
      int U=tmp/RES;
      int V=tmp-U*RES;
      int ui=U-half;
      int vi=V-half;
      float ugu=ui*du;
      float vgu=vi*du;
      float phase=TWO_PI*(ugu*lp + vgu*mp);
      float2 ej=cexpj(phase);
      acc=cadd(acc, cmul(shV[k],ej));
    }
    __syncthreads();
  }
  Cseg[pix]=acc.x;
}

template<int CHUNK_KEYS>
__global__ void recon_seg_blockage(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const int* __restrict__ offsets,
    const int* __restrict__ idx_sorted,
    const float2* __restrict__ Viss,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int nuniq,
    int RES, int half, float du,
    float fa, float fb,
    float cosphi,
    int occ_mode,
    float* __restrict__ Cseg,
    uint32_t* __restrict__ Wseg)
{
  __shared__ int shK[CHUNK_KEYS];
  __shared__ int shO0[CHUNK_KEYS];
  __shared__ int shO1[CHUNK_KEYS];

  long long pix=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(pix>=n_chunk) return;

  float lp0=l[pix], mp0=m[pix], np0=n[pix];
  float lp=lp0 + fa*np0;
  float mp=mp0 + fb*np0;

  float2 acc=make_float2(0,0);
  uint32_t wacc=0;
  const float TWO_PI=6.2831853071795864769f;

  for(int base=0;base<nuniq;base+=CHUNK_KEYS){
    int t=threadIdx.x;
    if(t<CHUNK_KEYS){
      int q=base+t;
      if(q<nuniq){
        shK[t]=keys_unique[q];
        shO0[t]=offsets[q];
        shO1[t]=offsets[q+1];
      } else {
        shK[t]=0;
        shO0[t]=0;
        shO1[t]=0;
      }
    }
    __syncthreads();

    #pragma unroll
    for(int kk=0; kk<CHUNK_KEYS; kk++){
      int key=shK[kk];
      if(key==0) continue;
      int s0=shO0[kk], s1=shO1[kk];
      int L=s1-s0;
      if(L<=0) continue;

      float2 sumV=make_float2(0,0);
      int cnt=0;

      auto handle_one = [&](int pos){
        int bi=idx_sorted[pos];
        float c1=(lp0*x1[bi] + mp0*y1[bi] + np0*z1[bi]) * invn1[bi];
        float c2=(lp0*x2[bi] + mp0*y2[bi] + np0*z2[bi]) * invn2[bi];
        if(c1>=cosphi && c2>=cosphi){
          float2 vv=Viss[bi];
          if(isfinite(vv.x) && isfinite(vv.y)){
            sumV.x+=vv.x;
            sumV.y+=vv.y;
            cnt++;
          }
        }
      };

      if(occ_mode==0){
        for(int pos=s0; pos<s1; pos++) handle_one(pos);
      } else if(occ_mode==1){
        handle_one(s0);
      } else if(occ_mode==2){
        handle_one(s0 + (L>>1));
      } else {
        handle_one(s0);
        handle_one(s0 + (L/3));
        handle_one(s0 + (2*L/3));
        handle_one(s1-1);
      }

      if(cnt>0){
        float invc=1.0f/(float)cnt;
        float2 vavg=make_float2(sumV.x*invc, sumV.y*invc);

        int tmp=key-1;
        int U=tmp/RES;
        int V=tmp-U*RES;
        int ui=U-half;
        int vi=V-half;
        float ugu=ui*du;
        float vgu=vi*du;

        float phase=TWO_PI*(ugu*lp + vgu*mp);
        float2 ej=cexpj(phase);
        acc=cadd(acc, cmul(vavg,ej));
        wacc += 1;
      }
    }
    __syncthreads();
  }

  Cseg[pix]=acc.x;
  Wseg[pix]=wacc;
}

__global__ void add_inplace(float* dst,const float* src,long long n){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}
__global__ void add_inplace_u32(uint32_t* dst,const uint32_t* src,long long n){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]+=src[i];
}
__global__ void scale_inplace(float* dst,long long n,float inv){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) dst[i]*=inv;
}
__global__ void normalize_by_weight(float* C,const uint32_t* W,long long n){
  long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n){
    uint32_t w=W[i];
    C[i]= (w>0)? (C[i]/(float)w) : 0.0f;
  }
}

static void uvw_stats(const float* u,const float* v,const float* w,int n,
                      float &umin,float &umax,float &vmin,float &vmax,float &wmin,float &wmax){
  umin=vmin=wmin=+INFINITY;
  umax=vmax=wmax=-INFINITY;
  for(int i=0;i<n;i++){
    float uu=u[i], vv=v[i], ww=w[i];
    if(uu<umin) umin=uu; if(uu>umax) umax=uu;
    if(vv<vmin) vmin=vv; if(vv>vmax) vmax=vv;
    if(ww<wmin) wmin=ww; if(ww>wmax) wmax=ww;
  }
}

static void compute_fa_fb_host_matlab_no_intercept(const float* u,const float* v,const float* w,int n,float &fa,float &fb,double &denom_out){
  long double su2=0, sv2=0, suv=0, suw=0, svw=0;
  for(int i=0;i<n;i++){
    long double uu=(long double)u[i];
    long double vv=(long double)v[i];
    long double ww=(long double)w[i];
    su2 += uu*uu;
    sv2 += vv*vv;
    suv += uu*vv;
    suw += uu*ww;
    svw += vv*ww;
  }
  long double denom = su2*sv2 - suv*suv;
  denom_out = (double)denom;
  long double scale=(su2*sv2>0)? (su2*sv2) : 1.0L;
  long double eps=1e-24L*scale;
  if(!std::isfinite((double)denom) || fabsl(denom)<eps){
    long double sign=(denom>=0)?1.0L:-1.0L;
    denom = sign*eps;
  }
  long double a=(sv2*suw - suv*svw)/denom;
  long double b=(su2*svw - suv*suw)/denom;
  if(!std::isfinite((double)a) || !std::isfinite((double)b)){
    a=0;
    b=0;
  }
  fa=(float)a;
  fb=(float)b;
}

struct GpuCtx {
  int dev=0;
  cudaStream_t stream=nullptr;
  long long pix0=0,pix1=0,n_chunk=0;

  float* d_B=nullptr;
  float* d_theta=nullptr;
  float* d_phi=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;

  int N=0;          // full-day baseline count
  int N_half=0;     // max segment half-baseline count
  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_x1=nullptr; float* d_y1=nullptr; float* d_z1=nullptr;
  float* d_x2=nullptr; float* d_y2=nullptr; float* d_z2=nullptr;
  float* d_invn1=nullptr; float* d_invn2=nullptr;

  float2* d_Vpart=nullptr;   // max_segN_half
  float2* d_Viss=nullptr;    // max_segN

  // recon accumulators
  float* d_Cacc=nullptr;
  float* d_Cseg=nullptr;
  uint32_t* d_Wacc=nullptr;
  uint32_t* d_Wseg=nullptr;

  // pinned host buffers
  float2* h_Vpart=nullptr;   // max_segN_half
  float* h_chunk=nullptr;
};

struct BaselineBroadcastInfo {
  int gen_dev = 0;
  std::vector<int> peer_ok;
  bool need_host_stage = false;
};

static void query_peer_access(BaselineBroadcastInfo& info, int gen_dev, const std::vector<GpuCtx>& ctx){
  info.gen_dev = gen_dev;
  info.peer_ok.assign(ctx.size(), 0);
  info.need_host_stage = false;

  for(size_t gi=0; gi<ctx.size(); ++gi){
    int dev = ctx[gi].dev;
    if(dev == gen_dev){
      info.peer_ok[gi] = 1;
      continue;
    }
    int can = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, dev, gen_dev));
    info.peer_ok[gi] = can;
    if(can){
      CHECK_CUDA(cudaSetDevice(dev));
      cudaError_t e = cudaDeviceEnablePeerAccess(gen_dev, 0);
      if(e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
      CHECK_CUDA(cudaGetLastError());
    }else{
      info.need_host_stage = true;
    }
  }
  CHECK_CUDA(cudaSetDevice(gen_dev));
  for(size_t gi=0; gi<ctx.size(); ++gi){
    int dev = ctx[gi].dev;
    if(dev == gen_dev) continue;
    int can = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can, gen_dev, dev));
    if(can){
      cudaError_t e = cudaDeviceEnablePeerAccess(dev, 0);
      if(e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CHECK_CUDA(e);
      CHECK_CUDA(cudaGetLastError());
    }
  }
}

static void broadcast_segment_to_gpus(const OrbitGenCtx& gen,
                                      const BaselineBroadcastInfo& info,
                                      std::vector<GpuCtx>& ctx,
                                      int segN,
                                      bool have_host_xyz)
{
  #pragma omp parallel for num_threads(64)
  for(int gi=0; gi<(int)ctx.size(); ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    cudaStream_t stream = ctx[gi].stream;

    auto copy_arr = [&](float* dst, const float* src_dev, const float* src_host){
      size_t bytes = (size_t)segN * sizeof(float);
      if(ctx[gi].dev == gen.dev){
        CHECK_CUDA(cudaMemcpyAsync(dst, src_dev, bytes, cudaMemcpyDeviceToDevice, stream));
      }else if(info.peer_ok[gi]){
        CHECK_CUDA(cudaMemcpyPeerAsync(dst, ctx[gi].dev, src_dev, gen.dev, bytes, stream));
      }else{
        CHECK_CUDA(cudaMemcpyAsync(dst, src_host, bytes, cudaMemcpyHostToDevice, stream));
      }
    };

    copy_arr(ctx[gi].d_u,  gen.d_u,  gen.h_u);
    copy_arr(ctx[gi].d_v,  gen.d_v,  gen.h_v);
    copy_arr(ctx[gi].d_w,  gen.d_w,  gen.h_w);
    copy_arr(ctx[gi].d_x1, gen.d_x1, gen.h_x1);
    copy_arr(ctx[gi].d_y1, gen.d_y1, gen.h_y1);
    copy_arr(ctx[gi].d_z1, gen.d_z1, gen.h_z1);
    copy_arr(ctx[gi].d_x2, gen.d_x2, gen.h_x2);
    copy_arr(ctx[gi].d_y2, gen.d_y2, gen.h_y2);
    copy_arr(ctx[gi].d_z2, gen.d_z2, gen.h_z2);

    int b2 = 256, g2 = (segN + b2 - 1) / b2;
    invnorm3_kernel<<<g2,b2,0,stream>>>(ctx[gi].d_x1, ctx[gi].d_y1, ctx[gi].d_z1, ctx[gi].d_invn1, segN);
    invnorm3_kernel<<<g2,b2,0,stream>>>(ctx[gi].d_x2, ctx[gi].d_y2, ctx[gi].d_z2, ctx[gi].d_invn2, segN);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
  }
}

int main(int argc,char** argv){
  std::string btag=get_arg(argc,argv,"--btag","10M");
  int nside=to_int(get_arg(argc,argv,"--nside","0"),0);
  int day=to_int(get_arg(argc,argv,"--day","1"),1);
  int segs=to_int(get_arg(argc,argv,"--segs","0"),0);

  int blockage=to_int(get_arg(argc,argv,"--blockage","1"),1);
  int occ_mode=to_int(get_arg(argc,argv,"--occ_mode","2"),2);
  int write_viss=to_int(get_arg(argc,argv,"--write_viss","1"),1);
  int write_baseline_txt=to_int(get_arg(argc,argv,"--write_baseline_txt","0"),0);
  std::string B_mode=get_arg(argc,argv,"--B_mode","auto");
  int gen_gpu_index=to_int(get_arg(argc,argv,"--gen_gpu_index","0"),0);
  uint64_t orbit_seed=to_u64(get_arg(argc,argv,"--orbit_seed","12345"), 12345ULL);

  std::string in_dir=norm_dir(get_arg(argc,argv,"--in_dir",""));
  std::string sky_dir=norm_dir(get_arg(argc,argv,"--sky_dir",""));
  std::string out_dir=get_arg(argc,argv,"--out_dir","./out/");
  std::string gpus_s=get_arg(argc,argv,"--gpus","0");

  int uvw_max=to_int(get_arg(argc,argv,"--uvw_max","4500000"),4500000);
  float B_clip=to_float(get_arg(argc,argv,"--B_clip","500000"),500000.0f);
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

    if(in_dir.empty()){
    if(btag=="1M") in_dir="./earth_1Mhz_cuda";
    else if(btag=="10M") in_dir="./earth_10Mhz_cuda";
    else in_dir="./earth_30Mhz_cuda";
  }
  if(sky_dir.empty()) sky_dir=(btag=="1M")? "./earth_1Mhz" : (btag=="10M"? "./earth_10Mhz":"./earth_30Mhz");
  if(!out_dir.empty() && out_dir.back()!='/') out_dir.push_back('/');
  ensure_dir(out_dir);

  auto gpus=parse_gpus(gpus_s);
  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  for(int d: gpus){
    if(d<0 || d>=devCount){
      std::cerr<<"ERROR gpu id "<<d<<" out of range "<<devCount<<"\n";
      return 1;
    }
  }
  int G=(int)gpus.size();

  long long npix=12LL*(long long)nside*(long long)nside;

  std::cout<<"btag="<<btag<<" nside="<<nside<<" npix="<<npix<<" day="<<day<<" segs="<<segs<<"\n";
  std::cout<<"blockage="<<blockage<<" occ_mode="<<occ_mode
           <<"  [Viss stage: blockage=ON, phase_correct=ON]\n";
  std::cout<<"in_dir="<<in_dir<<" sky_dir="<<sky_dir<<" out_dir="<<out_dir
           <<" gpus="<<gpus_s<<" uvw_max="<<uvw_max<<" B_clip="<<B_clip<<"\n";

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
    std::cerr << "ERROR gen_gpu_index out of range: " << gen_gpu_index << " / " << G << "\n";
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
    ofsViss << "0 0\n";
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

    std::cout<<"[seg "<<(s+1)<<"/"<<segs<<"] t0="<<t0
             <<" segN="<<segN
             <<" segN_half="<<segN_half
             <<" VissTime="<<t_vseg_s<<"s"
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
    if(ctx[gi].d_phi) cudaFree(ctx[gi].d_phi);
    if(ctx[gi].d_theta) cudaFree(ctx[gi].d_theta);
    if(ctx[gi].d_B) cudaFree(ctx[gi].d_B);

    if(ctx[gi].stream) cudaStreamDestroy(ctx[gi].stream);
  }

  return 0;
}
