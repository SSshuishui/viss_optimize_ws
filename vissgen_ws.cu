// vissgen_recon_blockage_nside.cu
// Unified 1M/10M version with optional blockage in BOTH Viss-gen and Recon (MATLAB-style).
// - NO thrust
// - Multi-GPU pixel-chunk split for sky pixels (l,m,n,B) like your optimized Viss code
// - Viss built from half + conj symmetry, then phase corrected: Viss *= exp(-i*2*pi*w)
// - Recon follows MATLAB locg/unique/group-average pipeline.
//   * blockage=0: group-average over all baselines in group (fast path)
//   * blockage=1: group-average per pixel with visibility mask (VERY heavy but correct)
//
// Build:
//   nvcc -O3 --use_fast_math -lineinfo -Xcompiler -fopenmp -std=c++14 vissgen_recon_blockage_nside.cu -o vissgen_recon_blockage_nside
//
// Run example:
//   ./vissgen_recon_blockage_nside --btag=10M --blockage=0 --nside=4096 --start_day=432 --end_day=450 \
//       --in_dir=/data/zhaox/earth_10Mhz --out_dir=./out10M/ --gpus=0,1,2,3 --uvw_max=4500000

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
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
#include <cub/device/device_reduce.cuh>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(1); \
  } \
} while(0)

struct HostTimer {
  using clock = std::chrono::high_resolution_clock;
  clock::time_point t0;
  void tic() { t0 = clock::now(); }
  double toc_s() const {
    return std::chrono::duration_cast<std::chrono::duration<double>>(clock::now() - t0).count();
  }
};

static inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline int to_int(const std::string& s, int defv) {
  try { return std::stoi(s); } catch(...) { return defv; }
}
static inline long long to_ll(const std::string& s, long long defv) {
  try { return std::stoll(s); } catch(...) { return defv; }
}
static inline float to_float(const std::string& s, float defv) {
  try { return std::stof(s); } catch(...) { return defv; }
}
static inline std::vector<int> parse_gpus(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::stoi(tok));
  }
  if (out.empty()) out.push_back(0);
  return out;
}
static inline std::string norm_dir(std::string p) {
  while (!p.empty() && p.back() == '/') p.pop_back();
  return p;
}
static inline void ensure_dir(const std::string& path) {
  std::string cmd = "mkdir -p " + path;
  std::system(cmd.c_str());
}

// -------- robust loader: file may be headerless or first line is npix --------
static bool load_single_auto_np(const std::string& path, float* out, long long n_expected) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;

  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; }

  // parse first token
  char* p = line;
  float first = strtof(p, &p);

  long long i = 0;
  // if first token looks like header npix
  long long maybeN = (long long)llround((double)first);
  bool is_header = (std::llabs(maybeN - n_expected) == 0);

  if (!is_header) {
    out[i++] = first;
  }

  while (i < n_expected && fgets(line, sizeof(line), fp)) {
    char* q = line;
    out[i++] = strtof(q, &q);
  }
  fclose(fp);
  return (i == n_expected);
}

static bool load_quad_skip_first(
    const std::string& path,
    float* a, float* b, float* c, float* d,
    int maxN,
    int& outN
) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; } // skip header line

  int n = 0;
  while (n < maxN && fgets(line, sizeof(line), fp)) {
    char* p = line;
    a[n] = strtof(p, &p);
    b[n] = strtof(p, &p);
    c[n] = strtof(p, &p);
    d[n] = strtof(p, &p);
    n++;
  }
  fclose(fp);
  outN = n;
  return true;
}

static bool load_triplets_skip_first(
    const std::string& path,
    float* a, float* b, float* c,
    int maxN,
    int& outN
) {
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) return false;
  char line[256];
  if (!fgets(line, sizeof(line), fp)) { fclose(fp); return false; } // skip header line

  int n = 0;
  while (n < maxN && fgets(line, sizeof(line), fp)) {
    char* p = line;
    a[n] = strtof(p, &p);
    b[n] = strtof(p, &p);
    c[n] = strtof(p, &p);
    n++;
  }
  fclose(fp);
  outN = n;
  return true;
}

__device__ __forceinline__ void sincos_fast(float x, float* s, float* c) { __sincosf(x, s, c); }

static inline __device__ float2 cadd(float2 a, float2 b) { return make_float2(a.x+b.x, a.y+b.y); }
static inline __device__ float2 cmul(float2 a, float2 b) { return make_float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
static inline __device__ float2 cscale(float2 a, float s) { return make_float2(a.x*s, a.y*s); }
static inline __device__ float2 cexpj(float phase) { float s,c; __sincosf(phase,&s,&c); return make_float2(c,s); }

// ============= healpix -> lmn (chunk) =============
__global__ void healpix_lmn_from_theta_phi_chunk(
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    long long n_chunk
) {
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_chunk) return;

  float theta_val = theta[idx];
  float phi_val   = phi[idx];

  theta_val = (float)M_PI * 0.5f - theta_val;
  if (phi_val > (float)M_PI) phi_val -= 2.0f * (float)M_PI;
  phi_val = -phi_val;

  float st, ct, sp, cp;
  sincos_fast(theta_val, &st, &ct);
  sincos_fast(phi_val,   &sp, &cp);

  l[idx] = ct * cp;
  m[idx] = ct * sp;
  n[idx] = st;
}

// ============= Viss partial (NO blockage) =============
template<int TILE_PIX>
__global__ void viss_partial_noblockage(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    int amount,
    float2* __restrict__ Vpart
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= amount) return;

  float u0 = u[i], v0 = v[i], w0 = w[i];
  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for (long long base = 0; base < n_chunk; base += TILE_PIX) {
    int t0 = threadIdx.x;
    if (t0 < TILE_PIX) {
      long long p = base + t0;
      if (p < n_chunk) {
        sB[t0] = B[p];
        sL[t0] = l[p];
        sM[t0] = m[p];
        sN[t0] = n[p];
      } else {
        sB[t0]=0; sL[t0]=0; sM[t0]=0; sN[t0]=0;
      }
    }
    int t1 = threadIdx.x + blockDim.x;
    if (t1 < TILE_PIX) {
      long long p = base + t1;
      if (p < n_chunk) {
        sB[t1] = B[p];
        sL[t1] = l[p];
        sM[t1] = m[p];
        sN[t1] = n[p];
      } else {
        sB[t1]=0; sL[t1]=0; sM[t1]=0; sN[t1]=0;
      }
    }
    __syncthreads();

    int tileN = (int)min((long long)TILE_PIX, n_chunk - base);
    #pragma unroll 4
    for (int t=0;t<tileN;t++) {
      float lp=sL[t], mp=sM[t], npv=sN[t];
      float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
      float ang = k * phase;
      float s, c;
      sincos_fast(ang, &s, &c);
      float bp = sB[t];
      acc_re += bp * c;
      acc_im += bp * s;
    }
    __syncthreads();
  }

  Vpart[i] = make_float2(acc_re, acc_im);
}

// ============= Viss partial (WITH blockage: beta1<phi && beta2<phi) =============
template<int TILE_PIX>
__global__ void viss_partial_blockage(
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
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int amount,
    float cosphi,
    float2* __restrict__ Vpart
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= amount) return;

  float u0 = u[i], v0 = v[i], w0 = w[i];

  float x1i=x1[i], y1i=y1[i], z1i=z1[i];
  float x2i=x2[i], y2i=y2[i], z2i=z2[i];

  float invn1 = rsqrtf(x1i*x1i + y1i*y1i + z1i*z1i);
  float invn2 = rsqrtf(x2i*x2i + y2i*y2i + z2i*z2i);

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for (long long base = 0; base < n_chunk; base += TILE_PIX) {
    int t0 = threadIdx.x;
    if (t0 < TILE_PIX) {
      long long p = base + t0;
      if (p < n_chunk) {
        sB[t0] = B[p];
        sL[t0] = l[p];
        sM[t0] = m[p];
        sN[t0] = n[p];
      } else {
        sB[t0]=0; sL[t0]=0; sM[t0]=0; sN[t0]=0;
      }
    }
    int t1 = threadIdx.x + blockDim.x;
    if (t1 < TILE_PIX) {
      long long p = base + t1;
      if (p < n_chunk) {
        sB[t1] = B[p];
        sL[t1] = l[p];
        sM[t1] = m[p];
        sN[t1] = n[p];
      } else {
        sB[t1]=0; sL[t1]=0; sM[t1]=0; sN[t1]=0;
      }
    }
    __syncthreads();

    int tileN = (int)min((long long)TILE_PIX, n_chunk - base);
    #pragma unroll 4
    for (int t=0;t<tileN;t++) {
      float lp=sL[t], mp=sM[t], npv=sN[t];

      // cos(beta) = dot(lmn, xyz)/|xyz|
      float c1 = (lp*x1i + mp*y1i + npv*z1i) * invn1;
      float c2 = (lp*x2i + mp*y2i + npv*z2i) * invn2;

      if (c1 >= cosphi && c2 >= cosphi) {
        float phase = u0*lp + v0*mp + w0*(npv - 1.0f);
        float ang = k * phase;
        float s, c;
        sincos_fast(ang, &s, &c);
        float bp = sB[t];
        acc_re += bp * c;
        acc_im += bp * s;
      }
    }
    __syncthreads();
  }

  Vpart[i] = make_float2(acc_re, acc_im);
}

__global__ void build_viss_full_from_half(
    const float2* __restrict__ Vhalf,
    float2* __restrict__ Vfull,
    int amount
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= amount) return;
  float2 v = Vhalf[i];
  Vfull[i] = v;
  Vfull[i + amount] = make_float2(v.x, -v.y);
}

__global__ void phase_correct_viss(
    float2* __restrict__ Viss,
    const float* __restrict__ w,
    int uvw_index
) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= uvw_index) return;

  float ang = -2.0f * (float)M_PI * w[idx];
  float s, c;
  sincos_fast(ang, &s, &c);

  float a = Viss[idx].x;
  float b = Viss[idx].y;

  float re = a * c - b * s;
  float im = a * s + b * c;
  Viss[idx] = make_float2(re, im);
}

// ================= fa/fb (double accumulation) =================
struct Sum5 { double u2,v2,uv,uw,vw; };

__global__ void sum5_stage1(const float* __restrict__ u,
                            const float* __restrict__ v,
                            const float* __restrict__ w,
                            int n, Sum5* __restrict__ blockOut)
{
  Sum5 local{0,0,0,0,0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    double uu=u[i], vv=v[i], ww=w[i];
    local.u2 += uu*uu;
    local.v2 += vv*vv;
    local.uv += uu*vv;
    local.uw += uu*ww;
    local.vw += vv*ww;
  }
  __shared__ Sum5 sh[256];
  int tid = threadIdx.x;
  sh[tid] = local;
  __syncthreads();
  for (int off = blockDim.x/2; off>0; off>>=1) {
    if (tid < off) {
      sh[tid].u2 += sh[tid+off].u2;
      sh[tid].v2 += sh[tid+off].v2;
      sh[tid].uv += sh[tid+off].uv;
      sh[tid].uw += sh[tid+off].uw;
      sh[tid].vw += sh[tid+off].vw;
    }
    __syncthreads();
  }
  if (tid==0) blockOut[blockIdx.x] = sh[0];
}

__global__ void sum5_stage2(const Sum5* __restrict__ blockIn, int nb, Sum5* __restrict__ out)
{
  Sum5 acc{0,0,0,0,0};
  for (int i=threadIdx.x; i<nb; i+=blockDim.x) {
    acc.u2 += blockIn[i].u2;
    acc.v2 += blockIn[i].v2;
    acc.uv += blockIn[i].uv;
    acc.uw += blockIn[i].uw;
    acc.vw += blockIn[i].vw;
  }
  __shared__ Sum5 sh[256];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int off=blockDim.x/2; off>0; off>>=1) {
    if (threadIdx.x < off) {
      sh[threadIdx.x].u2 += sh[threadIdx.x+off].u2;
      sh[threadIdx.x].v2 += sh[threadIdx.x+off].v2;
      sh[threadIdx.x].uv += sh[threadIdx.x+off].uv;
      sh[threadIdx.x].uw += sh[threadIdx.x+off].uw;
      sh[threadIdx.x].vw += sh[threadIdx.x+off].vw;
    }
    __syncthreads();
  }
  if (threadIdx.x==0) *out = sh[0];
}

static inline void compute_fa_fb(const float* d_u, const float* d_v, const float* d_w, int n,
                                 float &fa, float &fb, cudaStream_t stream)
{
  int block=256;
  int grid=120;
  Sum5 *d_block=nullptr, *d_out=nullptr;
  CHECK_CUDA(cudaMalloc(&d_block, grid*sizeof(Sum5)));
  CHECK_CUDA(cudaMalloc(&d_out, sizeof(Sum5)));

  sum5_stage1<<<grid, block, 0, stream>>>(d_u, d_v, d_w, n, d_block);
  sum5_stage2<<<1, 256, 0, stream>>>(d_block, grid, d_out);

  Sum5 h;
  CHECK_CUDA(cudaMemcpyAsync(&h, d_out, sizeof(Sum5), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  double denom = (h.u2*h.v2 - h.uv*h.uv);
  fa = (float)((h.v2*h.uw - h.uv*h.vw)/denom);
  fb = (float)((h.u2*h.vw - h.uv*h.uw)/denom);

  cudaFree(d_block);
  cudaFree(d_out);
}

// ================= locg / grouping (CUB) =================
__global__ void iota_kernel(int* idx, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) idx[i] = i;
}

__global__ void build_locg(const float* __restrict__ u,
                           const float* __restrict__ v,
                           int* __restrict__ locg,
                           int n, float inv_du, int RES, int half)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int ui = __float2int_rn(u[i] * inv_du);
    int vi = __float2int_rn(v[i] * inv_du);
    int U = ui + half;
    int V = vi + half;
    if ((unsigned)U >= (unsigned)RES || (unsigned)V >= (unsigned)RES) locg[i]=0;
    else locg[i] = U*RES + V + 1; // 1-based
  }
}

__global__ void set_last_offset(int* offsets, int nuniq, int n) {
  if (blockIdx.x==0 && threadIdx.x==0) offsets[nuniq] = n;
}

__global__ void gather_viss_by_idx(const float2* __restrict__ viss_in,
                                   const int* __restrict__ idx_sorted,
                                   float2* __restrict__ viss_sorted,
                                   int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    int j = idx_sorted[i];
    viss_sorted[i] = viss_in[j];
  }
}

__global__ void compute_avg(float2* avg, const float2* sum, const int* counts, int nuniq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nuniq) {
    int c = counts[i];
    avg[i] = (c>0) ? make_float2(sum[i].x / c, sum[i].y / c) : make_float2(0,0);
  }
}

struct Float2SumOp {
  __host__ __device__ float2 operator()(const float2& a, const float2& b) const {
    return make_float2(a.x+b.x, a.y+b.y);
  }
};

// Build unique keys/counts/offsets, and optionally avg per key (blockage=0).
// All done on one "master device" (gi0), then broadcast results.
static inline void build_groups_on_device0(
  int dev,
  cudaStream_t stream,
  const float* d_u, const float* d_v,
  const float2* d_viss,
  int n,
  float du,
  int RES,
  int half,
  // outputs:
  int* &d_keys_unique,   // [nuniq]
  int* &d_counts,        // [nuniq]
  int* &d_offsets,       // [nuniq+1]
  int* &d_idx_sorted,    // [n]  (only needed when blockage=1)
  float2* &d_viss_avg,   // [nuniq] (only for blockage=0)
  int &h_nuniq,
  int blockage
) {
  CHECK_CUDA(cudaSetDevice(dev));

  float inv_du = 1.0f / du;

  int *d_locg=nullptr;
  int *d_idx=nullptr, *d_idx_tmp=nullptr;
  int *d_keys_in=nullptr, *d_keys_tmp=nullptr;

  CHECK_CUDA(cudaMalloc(&d_locg, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx,  n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx_tmp, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_keys_in, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_keys_tmp, n*sizeof(int)));

  // build locg + idx
  {
    int block=256, grid=(n+block-1)/block;
    build_locg<<<grid, block, 0, stream>>>(d_u, d_v, d_locg, n, inv_du, RES, half);
    iota_kernel<<<grid, block, 0, stream>>>(d_idx, n);
  }
  CHECK_CUDA(cudaMemcpyAsync(d_keys_in, d_locg, n*sizeof(int), cudaMemcpyDeviceToDevice, stream));

  // sort pairs (keys, idx) by keys
  cub::DoubleBuffer<int> keys(d_keys_in, d_keys_tmp);
  cub::DoubleBuffer<int> vals(d_idx, d_idx_tmp);

  void* d_temp=nullptr;
  size_t temp_bytes=0;
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, keys, vals, n, 0, 8*sizeof(int), stream));
  CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, keys, vals, n, 0, 8*sizeof(int), stream));
  CHECK_CUDA(cudaFree(d_temp));

  int* d_keys_sorted = keys.Current();
  int* d_idx_s = vals.Current();

  // run-length => unique + counts
  d_keys_unique=nullptr;
  d_counts=nullptr;
  int* d_num_runs=nullptr;
  CHECK_CUDA(cudaMalloc(&d_keys_unique, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_counts, n*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_num_runs, sizeof(int)));

  d_temp=nullptr; temp_bytes=0;
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes,
        d_keys_sorted, d_keys_unique, d_counts, d_num_runs, n, stream));
  CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes,
        d_keys_sorted, d_keys_unique, d_counts, d_num_runs, n, stream));
  CHECK_CUDA(cudaFree(d_temp));

  CHECK_CUDA(cudaMemcpyAsync(&h_nuniq, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  cudaFree(d_num_runs);

  // offsets = exclusive_scan(counts), offsets[nuniq]=n
  CHECK_CUDA(cudaMalloc(&d_offsets, (h_nuniq+1)*sizeof(int)));
  d_temp=nullptr; temp_bytes=0;
  CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets, h_nuniq, stream));
  CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
  CHECK_CUDA(cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_counts, d_offsets, h_nuniq, stream));
  CHECK_CUDA(cudaFree(d_temp));
  set_last_offset<<<1,1,0,stream>>>(d_offsets, h_nuniq, n);

  // idx_sorted needed only when blockage=1
  if (blockage==1) {
    d_idx_sorted = nullptr;
    CHECK_CUDA(cudaMalloc(&d_idx_sorted, n*sizeof(int)));
    CHECK_CUDA(cudaMemcpyAsync(d_idx_sorted, d_idx_s, n*sizeof(int), cudaMemcpyDeviceToDevice, stream));
  } else {
    d_idx_sorted = nullptr;
  }

  // avg per key (blockage=0): ReduceByKey on (keys_sorted, viss_sorted)
  d_viss_avg = nullptr;
  if (blockage==0) {
    float2* d_viss_sorted=nullptr;
    CHECK_CUDA(cudaMalloc(&d_viss_sorted, n*sizeof(float2)));
    {
      int block=256, grid=(n+block-1)/block;
      gather_viss_by_idx<<<grid, block, 0, stream>>>(d_viss, d_idx_s, d_viss_sorted, n);
    }

    int* d_keys_out2=nullptr;
    float2* d_sum=nullptr;
    int* d_num2=nullptr;
    CHECK_CUDA(cudaMalloc(&d_keys_out2, h_nuniq*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_sum, h_nuniq*sizeof(float2)));
    CHECK_CUDA(cudaMalloc(&d_num2, sizeof(int)));

    d_temp=nullptr; temp_bytes=0;
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp, temp_bytes,
          d_keys_sorted, d_keys_out2,
          d_viss_sorted, d_sum,
          d_num2, Float2SumOp(), n, stream));
    CHECK_CUDA(cudaMalloc(&d_temp, temp_bytes));
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp, temp_bytes,
          d_keys_sorted, d_keys_out2,
          d_viss_sorted, d_sum,
          d_num2, Float2SumOp(), n, stream));
    CHECK_CUDA(cudaFree(d_temp));
    cudaFree(d_keys_out2);
    cudaFree(d_num2);
    cudaFree(d_viss_sorted);

    CHECK_CUDA(cudaMalloc(&d_viss_avg, h_nuniq*sizeof(float2)));
    {
      int block=256, grid=(h_nuniq+block-1)/block;
      compute_avg<<<grid, block, 0, stream>>>(d_viss_avg, d_sum, d_counts, h_nuniq);
    }
    cudaFree(d_sum);
  }

  CHECK_CUDA(cudaStreamSynchronize(stream));

  // cleanup temp
  cudaFree(d_locg);
  cudaFree(d_idx);
  cudaFree(d_idx_tmp);
  cudaFree(d_keys_in);
  cudaFree(d_keys_tmp);
}

// ================= invnorm for xyz =================
__global__ void invnorm3_kernel(const float* x, const float* y, const float* z, float* invn, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float a=x[i], b=y[i], c=z[i];
    invn[i] = rsqrtf(a*a + b*b + c*c);
  }
}

// ================= Recon kernels =================
// blockage=0: use per-key avg (scalar per key)
template<int CHUNK_KEYS>
__global__ void recon_C_real_keys_avg(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const float2* __restrict__ viss_avg,
    int nuniq,
    int RES,
    int half,
    float du,
    float fa,
    float fb,
    float* __restrict__ out_real
) {
  __shared__ int shK[CHUNK_KEYS];
  __shared__ float2 shV[CHUNK_KEYS];

  long long pix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= n_chunk) return;

  float lp = l[pix] + fa * n[pix];
  float mp = m[pix] + fb * n[pix];

  float2 acc = make_float2(0.f, 0.f);
  const float TWO_PI = 6.2831853071795864769f;

  for (int base=0; base<nuniq; base+=CHUNK_KEYS) {
    int t = threadIdx.x;
    if (t < CHUNK_KEYS) {
      int j = base + t;
      if (j < nuniq) { shK[t]=keys_unique[j]; shV[t]=viss_avg[j]; }
      else { shK[t]=0; shV[t]=make_float2(0,0); }
    }
    __syncthreads();

    #pragma unroll
    for (int k=0;k<CHUNK_KEYS;k++) {
      int key = shK[k];
      if (key==0) continue;
      int tmp = key - 1;
      int U = tmp / RES;
      int V = tmp - U*RES;
      int ui = U - half;
      int vi = V - half;
      float ugu = ui * du;
      float vgu = vi * du;

      float phase = TWO_PI * (ugu*lp + vgu*mp);
      float2 ej = cexpj(phase);
      acc = cadd(acc, cmul(shV[k], ej));
    }
    __syncthreads();
  }

  out_real[pix] = acc.x;
}

// blockage=1: per pixel, per key average only visible baselines in that group
template<int CHUNK_KEYS>
__global__ void recon_C_real_blockage(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const int* __restrict__ offsets,     // [nuniq+1]
    const int* __restrict__ idx_sorted,  // [uvw_n], maps sorted position -> baseline index
    const float2* __restrict__ viss,     // [uvw_n] original order
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int uvw_n,
    int nuniq,
    int RES,
    int half,
    float du,
    float fa,
    float fb,
    float cosphi,
    float* __restrict__ out_real
) {
  __shared__ int shK[CHUNK_KEYS];
  __shared__ int shO0[CHUNK_KEYS];
  __shared__ int shO1[CHUNK_KEYS];

  long long pix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (pix >= n_chunk) return;

  float lp0 = l[pix];
  float mp0 = m[pix];
  float np0 = n[pix];

  float lp = lp0 + fa * np0;
  float mp = mp0 + fb * np0;

  float2 acc = make_float2(0.f, 0.f);
  const float TWO_PI = 6.2831853071795864769f;

  for (int base=0; base<nuniq; base+=CHUNK_KEYS) {
    int t = threadIdx.x;
    if (t < CHUNK_KEYS) {
      int q = base + t;
      if (q < nuniq) {
        shK[t]  = keys_unique[q];
        shO0[t] = offsets[q];
        shO1[t] = offsets[q+1];
      } else {
        shK[t]=0; shO0[t]=0; shO1[t]=0;
      }
    }
    __syncthreads();

    #pragma unroll
    for (int kk=0; kk<CHUNK_KEYS; kk++) {
      int key = shK[kk];
      if (key==0) continue;

      int s0 = shO0[kk];
      int s1 = shO1[kk];

      float2 sumV = make_float2(0.f, 0.f);
      int cnt = 0;

      // visibility-gated average over baselines in group
      for (int pos=s0; pos<s1; pos++) {
        int bi = idx_sorted[pos];  // baseline index in original arrays
        float c1 = (lp0*x1[bi] + mp0*y1[bi] + np0*z1[bi]) * invn1[bi];
        float c2 = (lp0*x2[bi] + mp0*y2[bi] + np0*z2[bi]) * invn2[bi];
        if (c1 >= cosphi && c2 >= cosphi) {
          float2 v = viss[bi];
          sumV.x += v.x; sumV.y += v.y;
          cnt++;
        }
      }
      if (cnt > 0) {
        float invc = 1.0f / (float)cnt;
        float2 vavg = make_float2(sumV.x*invc, sumV.y*invc);

        int tmp = key - 1;
        int U = tmp / RES;
        int V = tmp - U*RES;
        int ui = U - half;
        int vi = V - half;
        float ugu = ui * du;
        float vgu = vi * du;

        float phase = TWO_PI * (ugu*lp + vgu*mp);
        float2 ej = cexpj(phase);
        acc = cadd(acc, cmul(vavg, ej));
      }
    }

    __syncthreads();
  }

  out_real[pix] = acc.x;
}

// ================= GPU context =================
struct GpuCtx {
  int dev = 0;
  cudaStream_t stream = nullptr;

  long long pix0 = 0;
  long long pix1 = 0;
  long long n_chunk = 0;

  // sky chunk
  float* d_B=nullptr;
  float* d_theta=nullptr;
  float* d_phi=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;

  // uvw and xyz arrays
  int UVW_MAX = 0;
  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_xyz1a=nullptr; float* d_xyz1b=nullptr; float* d_xyz1c=nullptr;
  float* d_xyz2a=nullptr; float* d_xyz2b=nullptr; float* d_xyz2c=nullptr;

  // invnorms (for blockage=1 recon)
  float* d_invn1=nullptr;
  float* d_invn2=nullptr;

  // Viss buffers
  float2* d_Vhalf=nullptr; // dev0 only
  float2* d_Viss=nullptr;  // [uvw_index] on each GPU

  // Recon outputs
  float* d_Creal=nullptr;

  // host pinned buffers
  float2* h_Vpart=nullptr; // [amount] pinned
  float*  h_chunk=nullptr; // pinned for output copy
};

// ================= main =================
int main(int argc, char** argv) {
  std::string btag = get_arg(argc, argv, "--btag", "10M");      // 1M / 10M
  int blockage = to_int(get_arg(argc, argv, "--blockage", "0"), 0);
  int nside = to_int(get_arg(argc, argv, "--nside", "0"), 0);
  int start_day = to_int(get_arg(argc, argv, "--start_day", "1"), 1);
  int end_day   = to_int(get_arg(argc, argv, "--end_day", "2"), 2);
  std::string in_dir  = norm_dir(get_arg(argc, argv, "--in_dir", ""));
  std::string out_dir = get_arg(argc, argv, "--out_dir", "");
  std::string gpus_s  = get_arg(argc, argv, "--gpus", "0");
  int uvw_max = to_int(get_arg(argc, argv, "--uvw_max", "450000"), 450000);

  if (btag != "1M" && btag != "10M") {
    std::cerr << "ERROR: --btag must be 1M or 10M\n";
    return 1;
  }
  if (nside <= 0) nside = (btag == "1M") ? 512 : 4096;

  long long npix = 12LL * (long long)nside * (long long)nside;
  if (npix <= 0) { std::cerr << "ERROR: invalid npix\n"; return 1; }

  if (in_dir.empty()) in_dir = (btag == "1M") ? "./earth_1Mhz" : "./earth_10Mhz";
  if (out_dir.empty()) out_dir = std::string("./out_") + btag + (blockage? "_blk/":"_noblk/");
  if (!out_dir.empty() && out_dir.back() != '/') out_dir.push_back('/');
  ensure_dir(out_dir);

  auto gpus = parse_gpus(gpus_s);
  int devCount=0;
  CHECK_CUDA(cudaGetDeviceCount(&devCount));
  for (int d : gpus) {
    if (d < 0 || d >= devCount) {
      std::cerr << "ERROR: gpu id " << d << " out of range, devCount=" << devCount << "\n";
      return 1;
    }
  }

  std::cout << "btag=" << btag << " blockage=" << blockage
            << " nside=" << nside << " npix=" << npix << "\n";
  std::cout << "days=[" << start_day << "," << end_day << "] in_dir=" << in_dir << " out_dir=" << out_dir << "\n";
  std::cout << "gpus=" << gpus_s << " uvw_max=" << uvw_max << "\n";

  // physical params
  float R=1737.1e3f, h=300e3f;
  float theta = asinf(R/(R+h));
  float phi = (float)M_PI - theta;
  float cosphi = cosf(phi);

  // recon params
  float frequency = (btag=="1M") ? 1e6f : 1e7f;
  float lamda = 3e8f / frequency;
  float bl_max = 100e3f;

  std::cout << "theta="<<theta<<" phi="<<phi<<" cosphi="<<cosphi<<" lamda="<<lamda<<"\n";

  HostTimer t_total; t_total.tic();

  // Load B/theta/phi (headerless or headered auto-detect)
  HostTimer t_io; t_io.tic();
  std::vector<float> hB(npix), hTheta(npix), hPhi(npix);

  // try these names first (your optimized code style), fallback to plain names
  std::string fB1 = in_dir + "/B_" + btag + ".txt";
  std::string fT1 = in_dir + "/theta_heal_" + btag + ".txt";
  std::string fP1 = in_dir + "/phi_heal_" + btag + ".txt";
  std::string fB2 = in_dir + "/B.txt";
  std::string fT2 = in_dir + "/theta_heal.txt";
  std::string fP2 = in_dir + "/phi_heal.txt";

  bool ok = load_single_auto_np(fB1, hB.data(), npix) && load_single_auto_np(fT1, hTheta.data(), npix) && load_single_auto_np(fP1, hPhi.data(), npix);
  if (!ok) {
    ok = load_single_auto_np(fB2, hB.data(), npix) && load_single_auto_np(fT2, hTheta.data(), npix) && load_single_auto_np(fP2, hPhi.data(), npix);
  }
  if (!ok) {
    std::cerr << "ERROR reading B/theta/phi for npix=" << npix << "\n";
    std::cerr << "Tried: " << fB1 << " / " << fT1 << " / " << fP1 << "\n";
    std::cerr << "Then : " << fB2 << " / " << fT2 << " / " << fP2 << "\n";
    return 1;
  }

  float s_scale = 4.0f * (float)M_PI / (float)npix;
  for (long long i=0;i<npix;i++) hB[i] *= s_scale;
  std::cout << "load B/theta/phi OK, s=" << s_scale << ", time=" << t_io.toc_s() << " s\n";

  int G = (int)gpus.size();
  std::vector<GpuCtx> ctx(G);
  long long chunk_size = (npix + G - 1) / G;

  // init per GPU
  HostTimer t_init; t_init.tic();
  #pragma omp parallel for num_threads(G)
  for (int gi=0; gi<G; ++gi) {
    int dev = gpus[gi];
    CHECK_CUDA(cudaSetDevice(dev));

    long long pix0 = (long long)gi * chunk_size;
    long long pix1 = std::min(npix, pix0 + chunk_size);
    long long n_chunk = std::max(0LL, pix1 - pix0);

    ctx[gi].dev = dev;
    ctx[gi].pix0 = pix0;
    ctx[gi].pix1 = pix1;
    ctx[gi].n_chunk = n_chunk;
    ctx[gi].UVW_MAX = uvw_max;

    CHECK_CUDA(cudaStreamCreateWithFlags(&ctx[gi].stream, cudaStreamNonBlocking));

    // sky chunk buffers
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_B,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_theta, (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_phi,   (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_l,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_m,     (size_t)n_chunk*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_n,     (size_t)n_chunk*sizeof(float)));

    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_B,     hB.data() + pix0,     (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_theta, hTheta.data() + pix0, (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));
    CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_phi,   hPhi.data() + pix0,   (size_t)n_chunk*sizeof(float), cudaMemcpyHostToDevice, ctx[gi].stream));

    // precompute lmn chunk
    const int BLOCK = 256;
    int grid = (int)((n_chunk + BLOCK - 1) / BLOCK);
    healpix_lmn_from_theta_phi_chunk<<<grid, BLOCK, 0, ctx[gi].stream>>>(
      ctx[gi].d_theta, ctx[gi].d_phi, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, n_chunk
    );
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(ctx[gi].stream));

    // theta/phi no longer needed on device
    CHECK_CUDA(cudaFree(ctx[gi].d_theta)); ctx[gi].d_theta=nullptr;
    CHECK_CUDA(cudaFree(ctx[gi].d_phi));   ctx[gi].d_phi=nullptr;

    // uvw & xyz buffers
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_u, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_v, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_w, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1a, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1b, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz1c, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2a, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2b, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_xyz2c, (size_t)uvw_max*sizeof(float)));

    // invnorm buffers (only used when blockage==1, but allocate anyway for simplicity)
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn1, (size_t)uvw_max*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_invn2, (size_t)uvw_max*sizeof(float)));

    // Viss buffers
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Viss, (size_t)uvw_max*sizeof(float2)));
    if (gi==0) CHECK_CUDA(cudaMalloc(&ctx[gi].d_Vhalf, (size_t)uvw_max*sizeof(float2)));

    // output chunk
    CHECK_CUDA(cudaMalloc(&ctx[gi].d_Creal, (size_t)n_chunk*sizeof(float)));

    // pinned buffers
    int max_amount = uvw_max / 2;
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_Vpart, (size_t)max_amount*sizeof(float2)));
    const int CHUNK = 1 << 20;
    CHECK_CUDA(cudaMallocHost(&ctx[gi].h_chunk, (size_t)CHUNK*sizeof(float)));
  }
  std::cout << "GPU init time=" << t_init.toc_s() << " s\n";

  // host arrays for uvw/xyz
  std::vector<float> hu(uvw_max), hv(uvw_max), hw(uvw_max), tmpf(uvw_max);
  std::vector<float> hxyz1a(uvw_max), hxyz1b(uvw_max), hxyz1c(uvw_max);
  std::vector<float> hxyz2a(uvw_max), hxyz2b(uvw_max), hxyz2c(uvw_max);
  std::vector<float2> hVhalf(uvw_max);
  std::vector<float2> hViss(uvw_max);

  static const size_t OUT_BUF_SZ = 8 << 20;
  static thread_local std::vector<char> outbuf(OUT_BUF_SZ);

  // per-day loop
  for (int day = start_day; day <= end_day; ++day) {
    HostTimer t_day; t_day.tic();

    // ---------- load uvw & xyz ----------
    HostTimer t_day_io; t_day_io.tic();

    std::string suf = "day" + btag + ".txt";
    std::string fuvw  = in_dir + "/updated_uvw" + std::to_string(day) + suf;
    std::string fxyz1 = in_dir + "/xyza" + std::to_string(day) + suf;
    std::string fxyz2 = in_dir + "/xyzb" + std::to_string(day) + suf;

    int uvw_index=0, xyz1_index=0, xyz2_index=0;

    if (!load_quad_skip_first(fuvw, hu.data(), hv.data(), hw.data(), tmpf.data(), uvw_max, uvw_index)) {
      std::cerr << "[day " << day << "] ERROR read " << fuvw << "\n";
      continue;
    }
    if (!load_triplets_skip_first(fxyz1, hxyz1a.data(), hxyz1b.data(), hxyz1c.data(), uvw_max, xyz1_index) ||
        !load_triplets_skip_first(fxyz2, hxyz2a.data(), hxyz2b.data(), hxyz2c.data(), uvw_max, xyz2_index)) {
      std::cerr << "[day " << day << "] ERROR read xyz files\n";
      continue;
    }
    if (uvw_index <= 0 || xyz1_index != uvw_index || xyz2_index != uvw_index) {
      std::cerr << "[day " << day << "] ERROR index mismatch uvw=" << uvw_index
                << " xyz1=" << xyz1_index << " xyz2=" << xyz2_index << "\n";
      continue;
    }
    if (uvw_index & 1) uvw_index -= 1;         // make even
    if (uvw_index <= 0) continue;
    int amount = uvw_index / 2;
    double day_io_s = t_day_io.toc_s();

    // ---------- H2D broadcast uvw/xyz ----------
    HostTimer t_h2d; t_h2d.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_u, hu.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_v, hv.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_w, hw.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1a, hxyz1a.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1b, hxyz1b.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz1c, hxyz1c.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2a, hxyz2a.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2b, hxyz2b.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_xyz2c, hxyz2c.data(), (size_t)uvw_index*sizeof(float), cudaMemcpyHostToDevice, stream));

      // invnorm (needed for blockage recon)
      int block=256, grid=(uvw_index+block-1)/block;
      invnorm3_kernel<<<grid, block, 0, stream>>>(ctx[gi].d_xyz1a, ctx[gi].d_xyz1b, ctx[gi].d_xyz1c, ctx[gi].d_invn1, uvw_index);
      invnorm3_kernel<<<grid, block, 0, stream>>>(ctx[gi].d_xyz2a, ctx[gi].d_xyz2b, ctx[gi].d_xyz2c, ctx[gi].d_invn2, uvw_index);

      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double h2d_s = t_h2d.toc_s();

    // ---------- Viss partial per GPU ----------
    HostTimer t_viss; t_viss.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      float2* d_Vpart = ctx[gi].d_Viss; // scratch

      constexpr int BASE_BLOCK = 128;
      int grid = (amount + BASE_BLOCK - 1) / BASE_BLOCK;
      constexpr int TILE_PIX = 256;
      size_t shmem = (size_t)TILE_PIX * 4 * sizeof(float);

      if (blockage==0) {
        viss_partial_noblockage<TILE_PIX><<<grid, BASE_BLOCK, shmem, stream>>>(
          ctx[gi].d_B, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
          ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
          amount,
          d_Vpart
        );
      } else {
        viss_partial_blockage<TILE_PIX><<<grid, BASE_BLOCK, shmem, stream>>>(
          ctx[gi].d_B, ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n, ctx[gi].n_chunk,
          ctx[gi].d_u, ctx[gi].d_v, ctx[gi].d_w,
          ctx[gi].d_xyz1a, ctx[gi].d_xyz1b, ctx[gi].d_xyz1c,
          ctx[gi].d_xyz2a, ctx[gi].d_xyz2b, ctx[gi].d_xyz2c,
          amount, cosphi,
          d_Vpart
        );
      }
      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].h_Vpart, d_Vpart, (size_t)amount*sizeof(float2), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // reduce partial Viss on host
    for (int i=0;i<amount;i++) {
      double re=0.0, im=0.0;
      for (int gi=0; gi<G; ++gi) {
        re += (double)ctx[gi].h_Vpart[i].x;
        im += (double)ctx[gi].h_Vpart[i].y;
      }
      hVhalf[i] = make_float2((float)re, (float)im);
    }
    double viss_s = t_viss.toc_s();

    // ---------- build full Viss + phase correct on device0, then broadcast ----------
    HostTimer t_vfix; t_vfix.tic();
    {
      int gi0 = 0;
      CHECK_CUDA(cudaSetDevice(ctx[gi0].dev));
      cudaStream_t stream = ctx[gi0].stream;

      CHECK_CUDA(cudaMemcpyAsync(ctx[gi0].d_Vhalf, hVhalf.data(), (size_t)amount*sizeof(float2), cudaMemcpyHostToDevice, stream));

      const int BLOCK = 256;
      int grid1 = (amount + BLOCK - 1) / BLOCK;
      build_viss_full_from_half<<<grid1, BLOCK, 0, stream>>>(ctx[gi0].d_Vhalf, ctx[gi0].d_Viss, amount);
      CHECK_CUDA(cudaPeekAtLastError());

      int grid2 = (uvw_index + BLOCK - 1) / BLOCK;
      phase_correct_viss<<<grid2, BLOCK, 0, stream>>>(ctx[gi0].d_Viss, ctx[gi0].d_w, uvw_index);
      CHECK_CUDA(cudaPeekAtLastError());

      CHECK_CUDA(cudaStreamSynchronize(stream));
      CHECK_CUDA(cudaMemcpyAsync(hViss.data(), ctx[gi0].d_Viss, (size_t)uvw_index*sizeof(float2), cudaMemcpyDeviceToHost, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // broadcast Viss to all gpus
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;
      CHECK_CUDA(cudaMemcpyAsync(ctx[gi].d_Viss, hViss.data(), (size_t)uvw_index*sizeof(float2), cudaMemcpyHostToDevice, stream));
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double vfix_s = t_vfix.toc_s();

    // ---------- Recon prep: fa/fb, RES/du, group build on device0 ----------
    HostTimer t_grp; t_grp.tic();

    float fa=0.f, fb=0.f;
    {
      // compute fa/fb on device0 using its u/v/w
      int gi0=0;
      CHECK_CUDA(cudaSetDevice(ctx[gi0].dev));
      compute_fa_fb(ctx[gi0].d_u, ctx[gi0].d_v, ctx[gi0].d_w, uvw_index, fa, fb, ctx[gi0].stream);
    }

    float lmax = std::sqrt(1.0f + fa*fa);
    float mmax = std::sqrt(1.0f + fb*fb);
    float lmmax = std::max(lmax, mmax);

    float temp_res = std::ceil(2.0f * bl_max / lamda * 2.0f * lmmax);
    int RES = (int)(temp_res + 2 + 1 - std::fmod(temp_res, 2.0f));
    int half = (RES - 1) / 2;
    float du = 2.0f * bl_max / lamda / (RES - 1);

    // group build outputs on device0
    int* d_keys_unique0=nullptr;
    int* d_counts0=nullptr;
    int* d_offsets0=nullptr;
    int* d_idx_sorted0=nullptr;
    float2* d_vavg0=nullptr;
    int nuniq=0;

    {
      int gi0=0;
      CHECK_CUDA(cudaSetDevice(ctx[gi0].dev));
      build_groups_on_device0(
        ctx[gi0].dev, ctx[gi0].stream,
        ctx[gi0].d_u, ctx[gi0].d_v,
        ctx[gi0].d_Viss,
        uvw_index,
        du, RES, half,
        d_keys_unique0, d_counts0, d_offsets0, d_idx_sorted0, d_vavg0, nuniq,
        (blockage==0 ? 0 : 1)
      );
    }

    // broadcast group artifacts
    // - blockage=0: keys_unique + viss_avg
    // - blockage=1: keys_unique + offsets + idx_sorted
    std::vector<int> h_keys_unique(nuniq);
    std::vector<float2> h_vavg(nuniq);
    std::vector<int> h_offsets(nuniq+1);
    std::vector<int> h_idx_sorted(uvw_index);

    {
      int gi0=0;
      CHECK_CUDA(cudaSetDevice(ctx[gi0].dev));
      CHECK_CUDA(cudaMemcpy(h_keys_unique.data(), d_keys_unique0, nuniq*sizeof(int), cudaMemcpyDeviceToHost));
      if (blockage==0) {
        CHECK_CUDA(cudaMemcpy(h_vavg.data(), d_vavg0, nuniq*sizeof(float2), cudaMemcpyDeviceToHost));
      } else {
        CHECK_CUDA(cudaMemcpy(h_offsets.data(), d_offsets0, (nuniq+1)*sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_idx_sorted.data(), d_idx_sorted0, uvw_index*sizeof(int), cudaMemcpyDeviceToHost));
      }
    }

    // allocate & upload to each gpu
    // (free after day)
    std::vector<int*> d_keys_unique(G, nullptr);
    std::vector<float2*> d_vavg(G, nullptr);
    std::vector<int*> d_offsets(G, nullptr);
    std::vector<int*> d_idx_sorted(G, nullptr);

    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      CHECK_CUDA(cudaMalloc(&d_keys_unique[gi], nuniq*sizeof(int)));
      CHECK_CUDA(cudaMemcpyAsync(d_keys_unique[gi], h_keys_unique.data(), nuniq*sizeof(int), cudaMemcpyHostToDevice, stream));

      if (blockage==0) {
        CHECK_CUDA(cudaMalloc(&d_vavg[gi], nuniq*sizeof(float2)));
        CHECK_CUDA(cudaMemcpyAsync(d_vavg[gi], h_vavg.data(), nuniq*sizeof(float2), cudaMemcpyHostToDevice, stream));
      } else {
        CHECK_CUDA(cudaMalloc(&d_offsets[gi], (nuniq+1)*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_idx_sorted[gi], uvw_index*sizeof(int)));
        CHECK_CUDA(cudaMemcpyAsync(d_offsets[gi], h_offsets.data(), (nuniq+1)*sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_idx_sorted[gi], h_idx_sorted.data(), uvw_index*sizeof(int), cudaMemcpyHostToDevice, stream));
      }

      CHECK_CUDA(cudaStreamSynchronize(stream));
    }

    // free device0 group buffers
    {
      CHECK_CUDA(cudaSetDevice(ctx[0].dev));
      cudaFree(d_keys_unique0);
      cudaFree(d_counts0);
      cudaFree(d_offsets0);
      if (d_idx_sorted0) cudaFree(d_idx_sorted0);
      if (d_vavg0) cudaFree(d_vavg0);
    }

    double grp_s = t_grp.toc_s();

    // ---------- Recon compute C per GPU chunk ----------
    HostTimer t_C; t_C.tic();
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      cudaStream_t stream = ctx[gi].stream;

      const int BLOCK = 256;
      int grid = (int)((ctx[gi].n_chunk + BLOCK - 1) / BLOCK);

      // clear output
      CHECK_CUDA(cudaMemsetAsync(ctx[gi].d_Creal, 0, (size_t)ctx[gi].n_chunk*sizeof(float), stream));

      if (blockage==0) {
        recon_C_real_keys_avg<256><<<grid, BLOCK, 0, stream>>>(
          ctx[gi].n_chunk,
          ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
          d_keys_unique[gi],
          d_vavg[gi],
          nuniq,
          RES, half, du, fa, fb,
          ctx[gi].d_Creal
        );
      } else {
        recon_C_real_blockage<64><<<grid, BLOCK, 0, stream>>>(
          ctx[gi].n_chunk,
          ctx[gi].d_l, ctx[gi].d_m, ctx[gi].d_n,
          d_keys_unique[gi],
          d_offsets[gi],
          d_idx_sorted[gi],
          ctx[gi].d_Viss,
          ctx[gi].d_xyz1a, ctx[gi].d_xyz1b, ctx[gi].d_xyz1c, ctx[gi].d_invn1,
          ctx[gi].d_xyz2a, ctx[gi].d_xyz2b, ctx[gi].d_xyz2c, ctx[gi].d_invn2,
          uvw_index,
          nuniq,
          RES, half, du, fa, fb, cosphi,
          ctx[gi].d_Creal
        );
      }

      CHECK_CUDA(cudaPeekAtLastError());
      CHECK_CUDA(cudaStreamSynchronize(stream));
    }
    double C_s = t_C.toc_s();

    // free broadcast group buffers
    #pragma omp parallel for num_threads(G)
    for (int gi=0; gi<G; ++gi) {
      CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
      if (d_keys_unique[gi]) cudaFree(d_keys_unique[gi]);
      if (d_vavg[gi]) cudaFree(d_vavg[gi]);
      if (d_offsets[gi]) cudaFree(d_offsets[gi]);
      if (d_idx_sorted[gi]) cudaFree(d_idx_sorted[gi]);
    }

    // ---------- write output ----------
    HostTimer t_w; t_w.tic();
    std::string out_path = out_dir + "C" + std::to_string(day) + "day" + btag + ".txt";
    std::ofstream ofs(out_path);
    if (!ofs.is_open()) {
      std::cerr << "[day " << day << "] ERROR open output: " << out_path << "\n";
    } else {
      ofs.rdbuf()->pubsetbuf(outbuf.data(), outbuf.size());
      const int CHUNK = 1 << 20;

      for (int gi=0; gi<G; ++gi) {
        CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
        for (long long off=0; off<ctx[gi].n_chunk; off += CHUNK) {
          int cur = (int)std::min((long long)CHUNK, ctx[gi].n_chunk - off);
          CHECK_CUDA(cudaMemcpy(ctx[gi].h_chunk, ctx[gi].d_Creal + off, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
          for (int i=0;i<cur;i++) ofs << ctx[gi].h_chunk[i] << "\n";
        }
      }
      ofs.close();
    }
    double w_s = t_w.toc_s();

    double day_s = t_day.toc_s();
    std::cout << "[day " << day << "] uvw_index=" << uvw_index
              << " io=" << day_io_s << "s"
              << " h2d=" << h2d_s << "s"
              << " Vpart+reduce=" << viss_s << "s"
              << " Vfix+bcast=" << vfix_s << "s"
              << " groups=" << grp_s << "s"
              << " C=" << C_s << "s"
              << " write=" << w_s << "s"
              << " total=" << day_s << "s\n";
  }

  // cleanup
  #pragma omp parallel for num_threads(G)
  for (int gi=0; gi<G; ++gi) {
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));

    if (ctx[gi].h_chunk) CHECK_CUDA(cudaFreeHost(ctx[gi].h_chunk));
    if (ctx[gi].h_Vpart) CHECK_CUDA(cudaFreeHost(ctx[gi].h_Vpart));

    if (ctx[gi].d_Creal) CHECK_CUDA(cudaFree(ctx[gi].d_Creal));
    if (ctx[gi].d_Viss)  CHECK_CUDA(cudaFree(ctx[gi].d_Viss));
    if (ctx[gi].d_Vhalf) CHECK_CUDA(cudaFree(ctx[gi].d_Vhalf));

    if (ctx[gi].d_invn2) CHECK_CUDA(cudaFree(ctx[gi].d_invn2));
    if (ctx[gi].d_invn1) CHECK_CUDA(cudaFree(ctx[gi].d_invn1));

    if (ctx[gi].d_xyz2c) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2c));
    if (ctx[gi].d_xyz2b) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2b));
    if (ctx[gi].d_xyz2a) CHECK_CUDA(cudaFree(ctx[gi].d_xyz2a));
    if (ctx[gi].d_xyz1c) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1c));
    if (ctx[gi].d_xyz1b) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1b));
    if (ctx[gi].d_xyz1a) CHECK_CUDA(cudaFree(ctx[gi].d_xyz1a));

    if (ctx[gi].d_w) CHECK_CUDA(cudaFree(ctx[gi].d_w));
    if (ctx[gi].d_v) CHECK_CUDA(cudaFree(ctx[gi].d_v));
    if (ctx[gi].d_u) CHECK_CUDA(cudaFree(ctx[gi].d_u));

    if (ctx[gi].d_n) CHECK_CUDA(cudaFree(ctx[gi].d_n));
    if (ctx[gi].d_m) CHECK_CUDA(cudaFree(ctx[gi].d_m));
    if (ctx[gi].d_l) CHECK_CUDA(cudaFree(ctx[gi].d_l));
    if (ctx[gi].d_phi) CHECK_CUDA(cudaFree(ctx[gi].d_phi));
    if (ctx[gi].d_theta) CHECK_CUDA(cudaFree(ctx[gi].d_theta));
    if (ctx[gi].d_B) CHECK_CUDA(cudaFree(ctx[gi].d_B));

    if (ctx[gi].stream) CHECK_CUDA(cudaStreamDestroy(ctx[gi].stream));
  }

  std::cout << "TOTAL time=" << t_total.toc_s() << " s\n";
  return 0;
}