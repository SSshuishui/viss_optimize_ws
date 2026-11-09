#pragma once

#include "common.hpp"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/iterator/counting_input_iterator.cuh>

__device__ __forceinline__ float2 cadd(float2 a,float2 b){ return make_float2(a.x+b.x,a.y+b.y); }
__device__ __forceinline__ float2 cmul(float2 a,float2 b){ return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ __forceinline__ float2 cexpj(float phase){ float s,c; __sincosf(phase,&s,&c); return make_float2(c,s); }
__device__ __forceinline__ int round_away_from_zero(float x){ return (x>=0.0f)? (int)floorf(x+0.5f) : (int)ceilf(x-0.5f); }
#ifndef OP_RECON_TASK_BL_VALUE
#define OP_RECON_TASK_BL_VALUE 128
#endif
#ifndef OP_RECON_PLAN_BL_CHUNK_VALUE
#define OP_RECON_PLAN_BL_CHUNK_VALUE 512
#endif
#ifndef OP_RECON_TASK_TILE_PIX_VALUE
#define OP_RECON_TASK_TILE_PIX_VALUE 256
#endif
#ifndef OP_RECON_DEBUG_CHECK_FINITE
#define OP_RECON_DEBUG_CHECK_FINITE 0
#endif
#ifndef OP_RECON_SKIP_ZERO_WEIGHTED_VISS
#define OP_RECON_SKIP_ZERO_WEIGHTED_VISS 0
#endif

static constexpr int OP_RECON_TASK_BL = OP_RECON_TASK_BL_VALUE;
static constexpr int OP_RECON_PLAN_BL_CHUNK = OP_RECON_PLAN_BL_CHUNK_VALUE;
static constexpr int OP_RECON_TASK_TILE_PIX = OP_RECON_TASK_TILE_PIX_VALUE;

__device__ __forceinline__ int halfidx_to_fullpos_dev(int ih){
  int group = ih / UNIQUE_BASELINES_PER_T;
  int j = ih - group * UNIQUE_BASELINES_PER_T;
  return group * SIGNED_BASELINES_PER_T + j;
}


template<int TILE_PIX>
__global__ void build_tile_cone_meta_kernel(const float* __restrict__ l,
                                            const float* __restrict__ m,
                                            const float* __restrict__ n,
                                            long long n_chunk,
                                            float* __restrict__ tile_cx,
                                            float* __restrict__ tile_cy,
                                            float* __restrict__ tile_cz,
                                            float* __restrict__ tile_cosA,
                                            float* __restrict__ tile_sinA,
                                            int ntile)
{
  int tid = blockIdx.x;
  if(tid >= ntile) return;
  long long p0 = (long long)tid * TILE_PIX;
  int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
  if(tileN <= 0) return;

  float sx=0.0f, sy=0.0f, sz=0.0f;
  for(int k=threadIdx.x; k<tileN; k+=blockDim.x){
    long long p = p0 + k;
    sx += l[p]; sy += m[p]; sz += n[p];
  }
  __shared__ float rsx[256], rsy[256], rsz[256];
  rsx[threadIdx.x]=sx; rsy[threadIdx.x]=sy; rsz[threadIdx.x]=sz;
  __syncthreads();
  for(int off=blockDim.x>>1; off>0; off>>=1){
    if(threadIdx.x < off){
      rsx[threadIdx.x] += rsx[threadIdx.x + off];
      rsy[threadIdx.x] += rsy[threadIdx.x + off];
      rsz[threadIdx.x] += rsz[threadIdx.x + off];
    }
    __syncthreads();
  }
  __shared__ float cx,cy,cz;
  if(threadIdx.x==0){
    float inv = rsqrtf(rsx[0]*rsx[0] + rsy[0]*rsy[0] + rsz[0]*rsz[0]);
    cx = rsx[0]*inv; cy = rsy[0]*inv; cz = rsz[0]*inv;
  }
  __syncthreads();

  float mind = 1.0f;
  for(int k=threadIdx.x; k<tileN; k+=blockDim.x){
    long long p = p0 + k;
    float d = cx*l[p] + cy*m[p] + cz*n[p];
    mind = fminf(mind, d);
  }
  __shared__ float rmin[256];
  rmin[threadIdx.x] = mind;
  __syncthreads();
  for(int off=blockDim.x>>1; off>0; off>>=1){
    if(threadIdx.x < off) rmin[threadIdx.x] = fminf(rmin[threadIdx.x], rmin[threadIdx.x + off]);
    __syncthreads();
  }
  if(threadIdx.x==0){
    float cA = fminf(1.0f, fmaxf(-1.0f, rmin[0]));
    tile_cx[tid]=cx; tile_cy[tid]=cy; tile_cz[tid]=cz;
    tile_cosA[tid]=cA;
    tile_sinA[tid]=sqrtf(fmaxf(0.0f, 1.0f - cA*cA));
  }
}


__global__ void build_nm1_kernel(const float* __restrict__ n,
                                 float* __restrict__ nm1,
                                 long long n_chunk)
{
  long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_chunk) nm1[idx] = n[idx] - 1.0f;
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_tilecone(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ nm1,
    long long n_chunk,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart);

__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc = fminf(1.0f, fmaxf(-1.0f, dotc));
  float sind = sqrtf(fmaxf(0.0f, 1.0f - dotc*dotc));
  float lower = dotc * cosA - sind * sinA;
  float upper = dotc * cosA + sind * sinA;
  all_vis = (lower >= cosphi);
  all_hid = (upper <  cosphi);
}


__device__ __forceinline__ void operator_classify_pair_tile_dirs(
    float dx1, float dy1, float dz1,
    float dx2, float dy2, float dz2,
    float cx, float cy, float cz,
    float cosA, float sinA, float cosphi,
    bool &all_visible, bool &all_hidden)
{
  bool vis1=false, hid1=false, vis2=false, hid2=false;
  classify_tile_cone(dx1*cx + dy1*cy + dz1*cz, cosA, sinA, cosphi, vis1, hid1);
  classify_tile_cone(dx2*cx + dy2*cy + dz2*cz, cosA, sinA, cosphi, vis2, hid2);
  all_hidden  = hid1 || hid2;
  all_visible = vis1 && vis2;
}

__device__ __forceinline__ bool operator_point_visible_pair_dirs(
    float dx1, float dy1, float dz1,
    float dx2, float dy2, float dz2,
    float lp, float mp, float npv, float cosphi)
{
  float c1 = lp*dx1 + mp*dy1 + npv*dz1;
  if(c1 < cosphi) return false;
  float c2 = lp*dx2 + mp*dy2 + npv*dz2;
  return (c2 >= cosphi);
}

__device__ __forceinline__ float operator_forward_phase(float ku, float kv, float kw,
                                                        float lp, float mp, float nm1)
{
  return fmaf(kw, nm1, fmaf(kv, mp, ku * lp));
}

__device__ __forceinline__ float operator_adjoint_phase(float u, float v, float w,
                                                        float lp, float mp, float npv)
{
  return 6.2831853071795864769f * fmaf(w, npv, fmaf(v, mp, u * lp));
}


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

__global__ void pix2lmn_ring_kernel(
    int nside,
    unsigned int base_ipix,
    int chunkN,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;

  const double PI = 3.141592653589793238462643383279502884;

  unsigned long long ipix = base_ipix + (unsigned int)tid;
  unsigned long long ipix1 = ipix + 1ull;  // HEALPix RING formula uses 1-based index

  unsigned long long ns = (unsigned long long)nside;
  unsigned long long nl2 = 2ull * ns;
  unsigned long long nl4 = 4ull * ns;
  unsigned long long ncap = 2ull * ns * (ns - 1ull);
  unsigned long long npix = 12ull * ns * ns;

  double z;
  double phi;

  if (ipix1 <= ncap) {
    // North polar cap
    int iring = (int)(0.5 * (1.0 + sqrt(1.0 + 2.0 * (double)ipix1)));
    unsigned long long iphi = ipix1 - 2ull * (unsigned long long)iring * (unsigned long long)(iring - 1);

    z = 1.0 - ((double)iring * (double)iring) / (3.0 * (double)ns * (double)ns);
    phi = ((double)iphi - 0.5) * PI / (2.0 * (double)iring);
  }
  else if (ipix1 <= nl2 * (5ull * ns + 1ull)) {
    // Equatorial region
    unsigned long long ip = ipix1 - ncap - 1ull;
    int iring = (int)(ip / nl4) + nside;
    unsigned long long iphi = (ip % nl4) + 1ull;

    double fodd = 0.5 * (1.0 + (double)((iring + nside) & 1));

    z = ((double)(2 * nside - iring)) * 2.0 / (3.0 * (double)nside);
    phi = ((double)iphi - fodd) * PI / (2.0 * (double)nside);
  }
  else {
    // South polar cap
    unsigned long long ip = npix - ipix1 + 1ull;
    int iring = (int)(0.5 * (1.0 + sqrt(1.0 + 2.0 * (double)ip)));
    unsigned long long iphi =
        4ull * (unsigned long long)iring + 1ull
        - (ip - 2ull * (unsigned long long)iring * (unsigned long long)(iring - 1));

    z = -1.0 + ((double)iring * (double)iring) / (3.0 * (double)ns * (double)ns);
    phi = ((double)iphi - 0.5) * PI / (2.0 * (double)iring);
  }

  double sintheta = sqrt(fmax(0.0, 1.0 - z * z));

  l[tid] = (float)(sintheta * cos(phi));
  m[tid] = (float)(sintheta * sin(phi));
  n[tid] = (float)z;
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

  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float x1i = 0.0f, y1i = 0.0f, z1i = 0.0f;
  float x2i = 0.0f, y2i = 0.0f, z2i = 0.0f;
  float in1 = 0.0f, in2 = 0.0f;

  if(active){
    u0 = u[ih]; v0 = v[ih]; w0 = w[ih];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[ih]; y1i = y1[ih]; z1i = z1[ih]; in1 = invn1[ih];
      x2i = x2[ih]; y2i = y2[ih]; z2i = z2[ih]; in2 = invn2[ih];
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

static inline void launch_build_tile_cone_meta_runtime(
    int tile_pix, cudaStream_t stream,
    const float* d_l, const float* d_m, const float* d_n, long long n_chunk,
    float* d_tile_cx, float* d_tile_cy, float* d_tile_cz,
    float* d_tile_cosA, float* d_tile_sinA, int ntile)
{
  if(tile_pix == 512){
    build_tile_cone_meta_kernel<512><<<ntile,256,0,stream>>>(d_l,d_m,d_n,n_chunk,d_tile_cx,d_tile_cy,d_tile_cz,d_tile_cosA,d_tile_sinA,ntile);
  }else{
    build_tile_cone_meta_kernel<256><<<ntile,256,0,stream>>>(d_l,d_m,d_n,n_chunk,d_tile_cx,d_tile_cy,d_tile_cz,d_tile_cosA,d_tile_sinA,ntile);
  }
}

template<bool DO_BLOCKAGE>
static inline void launch_operator_forward_halfsym_runtime(
    int tile_pix, int gridV, int blockV, size_t shmem, cudaStream_t stream,
    const float* d_B, const float* d_l, const float* d_m, const float* d_n, const float* d_nm1, long long n_chunk,
    const float* d_tile_cx, const float* d_tile_cy, const float* d_tile_cz,
    const float* d_tile_cosA, const float* d_tile_sinA, int ntile,
    const float* d_u, const float* d_v, const float* d_w,
    const float* d_x1, const float* d_y1, const float* d_z1,
    const float* d_x2, const float* d_y2, const float* d_z2,
    int N_half, float cosphi, float2* d_Vpart)
{
  if(tile_pix == 512){
    viss_partial_all_halfsym_tilecone<512,DO_BLOCKAGE><<<gridV,blockV,shmem,stream>>>(d_B,d_l,d_m,d_n,d_nm1,n_chunk,d_tile_cx,d_tile_cy,d_tile_cz,d_tile_cosA,d_tile_sinA,ntile,d_u,d_v,d_w,d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,N_half,cosphi,d_Vpart);
  }else{
    viss_partial_all_halfsym_tilecone<256,DO_BLOCKAGE><<<gridV,blockV,shmem,stream>>>(d_B,d_l,d_m,d_n,d_nm1,n_chunk,d_tile_cx,d_tile_cy,d_tile_cz,d_tile_cosA,d_tile_sinA,ntile,d_u,d_v,d_w,d_x1,d_y1,d_z1,d_x2,d_y2,d_z2,N_half,cosphi,d_Vpart);
  }
}

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_tilecone(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ nm1,
    long long n_chunk,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  float dx1 = 0.0f, dy1 = 0.0f, dz1 = 0.0f;
  float dx2 = 0.0f, dy2 = 0.0f, dz2 = 0.0f;

  if(active){
    u0 = u[ih]; v0 = v[ih]; w0 = w[ih];
    if constexpr (DO_BLOCKAGE){
      dx1 = x1[ih]; dy1 = y1[ih]; dz1 = z1[ih];
      dx2 = x2[ih]; dy2 = y2[ih]; dz2 = z2[ih];
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  float ku=0.0f, kv=0.0f, kw=0.0f;
  if(active){
    ku = k * u0;
    kv = k * v0;
    kw = k * w0;
  }

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sNM1 = sM + TILE_PIX;

  for(int tid=0; tid<ntile; ++tid){
    long long p0 = (long long)tid * TILE_PIX;
    int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
    for(int lane = threadIdx.x; lane < tileN; lane += blockDim.x){
      long long p = p0 + lane;
      sB[lane] = B[p];
      sL[lane] = l[p];
      sM[lane] = m[p];
      sNM1[lane] = nm1[p];
    }
    __syncthreads();

    bool all_visible = false;
    bool all_hidden  = false;
    if(active && DO_BLOCKAGE){
      float cx = tile_cx[tid], cy = tile_cy[tid], cz = tile_cz[tid];
      float cosA = tile_cosA[tid], sinA = tile_sinA[tid];
      operator_classify_pair_tile_dirs(dx1,dy1,dz1,dx2,dy2,dz2,cx,cy,cz,cosA,sinA,cosphi,all_visible,all_hidden);
    }

    if(active && !(DO_BLOCKAGE && all_hidden)) {
      if(!DO_BLOCKAGE || all_visible){
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], nm1v = sNM1[k0];
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          acc_re += bp*c;
          acc_im += bp*s;
        }
      }else{
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], nm1v = sNM1[k0];
          float npv = nm1v + 1.0f;
          float c1=(lp*dx1 + mp*dy1 + npv*dz1);
          float c2=(lp*dx2 + mp*dy2 + npv*dz2);
          if(c1<cosphi || c2<cosphi) continue;
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          acc_re += bp*c;
          acc_im += bp*s;
        }
      }
    }
    __syncthreads();
  }

  if(active) Vpart[ih] = make_float2(acc_re, acc_im);
}

__global__ void build_sat_raw_from_pos(const float* __restrict__ pos,
                                       float* __restrict__ satx,
                                       float* __restrict__ saty,
                                       float* __restrict__ satz,
                                       int tlen)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = tlen * SATNUM;
  if(idx >= n) return;
  int t = idx / SATNUM;
  int s = idx - t * SATNUM;
  int base = t * (3 * SATNUM);
  satx[idx] = pos[base + s];
  saty[idx] = pos[base + SATNUM + s];
  satz[idx] = pos[base + 2*SATNUM + s];
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
  std::vector<int> counts;         // half/full group counts
  std::vector<float2> viss_avg;    // blockage=0
  std::vector<int> offsets;        // blockage=1
  std::vector<int> idx_sorted;     // blockage=1
};


__global__ void build_locg_half_kernel(
    const float* __restrict__ u_full,
    const float* __restrict__ v_full,
    int* __restrict__ locg,
    int n_half,
    float inv_du,
    int RES,
    int half)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= n_half) return;
  int i = halfidx_to_fullpos_dev(ih);
  float uu = u_full[i];
  float vv = v_full[i];

  int U = (int)floorf(uu * inv_du + 0.5f) + half;
  int V = (int)floorf(vv * inv_du + 0.5f) + half;
  if(U < 0 || U >= RES || V < 0 || V >= RES) locg[ih] = 0;
  else locg[ih] = U * RES + V + 1;
}

static void build_groups_device0_half(
    int dev, cudaStream_t stream,
    const float* d_u_full, const float* d_v_full,
    const float2* d_viss_half,
    int n_half,
    float du, int RES, int half,
    int blockage,
    GroupArtifactsHost& hout)
{
  CHECK_CUDA(cudaSetDevice(dev));
  hout = GroupArtifactsHost{};
  if(n_half <= 0) return;

  int *d_locg=nullptr, *d_idx=nullptr, *d_keys_in=nullptr, *d_keys_tmp=nullptr, *d_idx_tmp=nullptr;
  CHECK_CUDA(cudaMalloc(&d_locg, n_half*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx,  n_half*sizeof(int)));

  int block=256, grid=(n_half+block-1)/block;
  build_locg_half_kernel<<<grid,block,0,stream>>>(d_u_full,d_v_full,d_locg,n_half,1.0f/du,RES,half);
  iota_kernel<<<grid,block,0,stream>>>(d_idx,n_half);

  CHECK_CUDA(cudaMalloc(&d_keys_in,  n_half*sizeof(int)));
  CHECK_CUDA(cudaMemcpyAsync(d_keys_in,d_locg,n_half*sizeof(int),cudaMemcpyDeviceToDevice,stream));
  CHECK_CUDA(cudaMalloc(&d_keys_tmp, n_half*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_idx_tmp,  n_half*sizeof(int)));

  void* d_temp=nullptr; size_t temp_bytes=0;
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,d_keys_in,d_keys_tmp,d_idx,d_idx_tmp,n_half,0,32,stream));
  CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
  CHECK_CUDA(cub::DeviceRadixSort::SortPairs(d_temp,temp_bytes,d_keys_in,d_keys_tmp,d_idx,d_idx_tmp,n_half,0,32,stream));
  CHECK_CUDA(cudaFree(d_temp));

  int* d_keys_sorted=d_keys_tmp;
  int* d_idx_sorted =d_idx_tmp;

  int* d_keys_unique=nullptr;
  int* d_counts=nullptr;
  int* d_num_runs=nullptr;
  CHECK_CUDA(cudaMalloc(&d_keys_unique, n_half*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_counts,      n_half*sizeof(int)));
  CHECK_CUDA(cudaMalloc(&d_num_runs, sizeof(int)));

  d_temp=nullptr; temp_bytes=0;
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp,temp_bytes,d_keys_sorted,d_keys_unique,d_counts,d_num_runs,n_half,stream));
  CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
  CHECK_CUDA(cub::DeviceRunLengthEncode::Encode(d_temp,temp_bytes,d_keys_sorted,d_keys_unique,d_counts,d_num_runs,n_half,stream));
  CHECK_CUDA(cudaFree(d_temp));

  int nuniq=0;
  CHECK_CUDA(cudaMemcpy(&nuniq, d_num_runs, sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaFree(d_num_runs));
  if(nuniq<=0){
    CHECK_CUDA(cudaFree(d_keys_unique));
    CHECK_CUDA(cudaFree(d_counts));
    CHECK_CUDA(cudaFree(d_locg));
    CHECK_CUDA(cudaFree(d_idx));
    CHECK_CUDA(cudaFree(d_idx_tmp));
    CHECK_CUDA(cudaFree(d_keys_in));
    CHECK_CUDA(cudaFree(d_keys_tmp));
    return;
  }

  hout.nuniq=nuniq;
  hout.keys_unique.assign(nuniq,0);
  hout.counts.assign(nuniq,0);
  CHECK_CUDA(cudaMemcpy(hout.keys_unique.data(), d_keys_unique, nuniq*sizeof(int), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(hout.counts.data(),      d_counts,      nuniq*sizeof(int), cudaMemcpyDeviceToHost));

  if(blockage==0){
    float2* d_sum=nullptr;
    CHECK_CUDA(cudaMalloc(&d_sum, nuniq*sizeof(float2)));

    int* d_keys_out2=nullptr;
    int* d_num2=nullptr;
    CHECK_CUDA(cudaMalloc(&d_keys_out2,nuniq*sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_num2,sizeof(int)));

    float2* d_viss_sorted=nullptr;
    CHECK_CUDA(cudaMalloc(&d_viss_sorted, n_half*sizeof(float2)));
    gather_viss_by_idx<<<grid,block,0,stream>>>(d_viss_half,d_idx_sorted,d_viss_sorted,n_half);

    d_temp=nullptr;
    temp_bytes=0;
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,temp_bytes,d_keys_sorted,d_keys_out2,d_viss_sorted,d_sum,d_num2,Float2AddOp(),n_half,stream));
    CHECK_CUDA(cudaMalloc(&d_temp,temp_bytes));
    CHECK_CUDA(cub::DeviceReduce::ReduceByKey(d_temp,temp_bytes,d_keys_sorted,d_keys_out2,d_viss_sorted,d_sum,d_num2,Float2AddOp(),n_half,stream));
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
    set_last_offset<<<1,1,0,stream>>>(d_offsets,nuniq,n_half);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    hout.offsets.assign(nuniq+1,0);
    hout.idx_sorted.assign(n_half,0);
    CHECK_CUDA(cudaMemcpy(hout.offsets.data(), d_offsets, (nuniq+1)*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hout.idx_sorted.data(), d_idx_sorted, n_half*sizeof(int), cudaMemcpyDeviceToHost));
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


// ----- recon kernels (half-grid mirror mode) -----
template<int CHUNK_KEYS>
__global__ void recon_seg_keys_avg_half(
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
  float acc=0.0f;
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
      float s,c; sincos_fast(phase,&s,&c);
      float2 vv=shV[k];
      acc += vv.x*c - vv.y*s; // Re(v * e^{j phase}), mirror half absorbed in normalization
    }
    __syncthreads();
  }
  Cseg[pix]=acc;
}

template<int CHUNK_KEYS>
__global__ void recon_seg_blockage_half(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const int* __restrict__ keys_unique,
    const int* __restrict__ offsets,
    const int* __restrict__ idx_sorted,
    const float2* __restrict__ Viss_half,
    const float* __restrict__ x1_full,
    const float* __restrict__ y1_full,
    const float* __restrict__ z1_full,
    const float* __restrict__ invn1_full,
    const float* __restrict__ x2_full,
    const float* __restrict__ y2_full,
    const float* __restrict__ z2_full,
    const float* __restrict__ invn2_full,
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

  float acc=0.0f;
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
        shK[t]=0; shO0[t]=0; shO1[t]=0;
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
        int bih=idx_sorted[pos];
        int bi=halfidx_to_fullpos_dev(bih);
        float c1=(lp0*x1_full[bi] + mp0*y1_full[bi] + np0*z1_full[bi]) * invn1_full[bi];
        float c2=(lp0*x2_full[bi] + mp0*y2_full[bi] + np0*z2_full[bi]) * invn2_full[bi];
        if(c1>=cosphi && c2>=cosphi){
          float2 vv=Viss_half[bih];
          if(isfinite(vv.x) && isfinite(vv.y)){
            sumV.x += vv.x;
            sumV.y += vv.y;
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
        float s,c; sincos_fast(phase,&s,&c);
        acc += vavg.x*c - vavg.y*s; // mirror half absorbed in normalization
        wacc += 1;
      }
    }
    __syncthreads();
  }

  Cseg[pix]=acc;
  Wseg[pix]=wacc;
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
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
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
        float c1=(lp0*x1[bi] + mp0*y1[bi] + np0*z1[bi]);
        float c2=(lp0*x2[bi] + mp0*y2[bi] + np0*z2[bi]);
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


struct GpuSegSlot {
  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_x1=nullptr; float* d_y1=nullptr; float* d_z1=nullptr;
  float* d_x2=nullptr; float* d_y2=nullptr; float* d_z2=nullptr;
  float* d_invn1=nullptr; float* d_invn2=nullptr;

  float2* d_Vpart=nullptr;
  float2* d_Viss=nullptr;
  float2* h_Vpart=nullptr;

  cudaEvent_t bcast_start=nullptr;
  cudaEvent_t ready=nullptr;
  cudaEvent_t viss_start=nullptr;
  cudaEvent_t viss_stop=nullptr;
  cudaEvent_t collect_ready=nullptr;
  cudaEvent_t reduce_start=nullptr;
  cudaEvent_t reduce_done=nullptr;
  cudaEvent_t scatter_done=nullptr;
  cudaEvent_t host_viss_ready=nullptr;
  cudaEvent_t pairw_ready=nullptr;
  int segN=0;
  int segN_half=0;
  int tlen=0;
};

struct OperatorPlanBuffer {
  unsigned char* d_vv_flags=nullptr;
  unsigned char* d_mixed_flags=nullptr;
  int* d_vv_tasks=nullptr;
  int* d_mixed_tasks=nullptr;
  int* d_num_vv=nullptr;
  int* d_num_mixed=nullptr;
  int* h_num_vv=nullptr;
  int* h_num_mixed=nullptr;
  void* d_select_temp_vv=nullptr;
  void* d_select_temp_mixed=nullptr;
  size_t select_temp_vv_bytes=0;
  size_t select_temp_mixed_bytes=0;
  cudaEvent_t build_start=nullptr;
  cudaEvent_t ready=nullptr;
  cudaEvent_t compute_done=nullptr;
  int last_b0=0;
  int last_chunk_n=0;
};

struct ReconGpuLoadStats {
  int gpu_index = -1;
  int gpu_dev = -1;
  int chunks = 0;
  int zero_task_chunks = 0;
  int vv_only_chunks = 0;
  int mixed_chunks = 0;
  long long vv_tasks = 0;
  long long mixed_tasks = 0;
  double host_plan_wait_s = 0.0;
  double gpu_plan_build_s = 0.0;
  double run_wall_s = 0.0;
  double sync_wait_s = 0.0;
  double max_chunk_plan_wait_s = 0.0;
  int max_chunk_plan_wait_idx = -1;
};

static inline void accumulate_recon_gpu_load_stats(ReconGpuLoadStats& dst, const ReconGpuLoadStats& src){
  dst.chunks += src.chunks;
  dst.zero_task_chunks += src.zero_task_chunks;
  dst.vv_only_chunks += src.vv_only_chunks;
  dst.mixed_chunks += src.mixed_chunks;
  dst.vv_tasks += src.vv_tasks;
  dst.mixed_tasks += src.mixed_tasks;
  dst.host_plan_wait_s += src.host_plan_wait_s;
  dst.gpu_plan_build_s += src.gpu_plan_build_s;
  dst.run_wall_s += src.run_wall_s;
  dst.sync_wait_s += src.sync_wait_s;
  if(src.max_chunk_plan_wait_s > dst.max_chunk_plan_wait_s){
    dst.max_chunk_plan_wait_s = src.max_chunk_plan_wait_s;
    dst.max_chunk_plan_wait_idx = src.max_chunk_plan_wait_idx;
  }
}

struct GpuCtx {
  int dev=0;
  cudaStream_t compute_stream=nullptr;
  cudaStream_t xfer_stream=nullptr;
  cudaStream_t reduce_stream=nullptr;
  long long pix0=0,pix1=0,n_chunk=0;
  long long recon_n_chunk=0;

  float* d_B=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;
  float* d_nm1=nullptr;
  float* d_l_recon=nullptr;
  float* d_m_recon=nullptr;
  float* d_n_recon=nullptr;
  int ntile_viss=0;
  int ntile_recon=0;
  float* d_tile_cx_viss=nullptr;
  float* d_tile_cy_viss=nullptr;
  float* d_tile_cz_viss=nullptr;
  float* d_tile_cosA_viss=nullptr;
  float* d_tile_sinA_viss=nullptr;
  float* d_tile_cx_recon=nullptr;
  float* d_tile_cy_recon=nullptr;
  float* d_tile_cz_recon=nullptr;
  float* d_tile_cosA_recon=nullptr;
  float* d_tile_sinA_recon=nullptr;

  OperatorPlanBuffer recon_plan[2];

  int N=0;
  int N_half=0;
  GpuSegSlot slots[2];

  float* d_Cacc=nullptr;
  float* d_Cseg=nullptr;
  uint32_t* d_Wacc=nullptr;
  uint32_t* d_Wseg=nullptr;

  float* h_chunk=nullptr;
};

struct BaselineBroadcastInfo {
  int gen_dev = 0;
  std::vector<int> peer_ok;
  bool need_host_stage = false;
};

struct ReducerExchangeInfo {
  int reducer_dev = 0;
  std::vector<int> collect_peer_ok;
  std::vector<int> scatter_peer_ok;
  bool need_host_stage_collect = false;
  bool need_host_stage_scatter = false;
};

static void try_enable_peer_access_between_visible_gpus(const std::vector<GpuCtx>& ctx){
  for(size_t i=0; i<ctx.size(); ++i){
    for(size_t j=0; j<ctx.size(); ++j){
      if(i == j) continue;
      int can = 0;
      CHECK_CUDA(cudaDeviceCanAccessPeer(&can, ctx[i].dev, ctx[j].dev));
      if(!can) continue;
      CHECK_CUDA(cudaSetDevice(ctx[i].dev));
      cudaError_t st = cudaDeviceEnablePeerAccess(ctx[j].dev, 0);
      if(st != cudaSuccess && st != cudaErrorPeerAccessAlreadyEnabled){
        CHECK_CUDA(st);
      }
      if(st == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
    }
  }
}

static void query_reducer_exchange(ReducerExchangeInfo& info, int reducer_dev, const std::vector<GpuCtx>& ctx){
  info.reducer_dev = reducer_dev;
  info.collect_peer_ok.assign(ctx.size(), 0);
  info.scatter_peer_ok.assign(ctx.size(), 0);
  info.need_host_stage_collect = false;
  info.need_host_stage_scatter = false;
  for(size_t gi=0; gi<ctx.size(); ++gi){
    int dev = ctx[gi].dev;
    if(dev == reducer_dev){
      info.collect_peer_ok[gi] = 1;
      info.scatter_peer_ok[gi] = 1;
      continue;
    }
    int can_collect = 0;
    int can_scatter = 0;
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_collect, reducer_dev, dev));
    CHECK_CUDA(cudaDeviceCanAccessPeer(&can_scatter, dev, reducer_dev));
    info.collect_peer_ok[gi] = can_collect;
    info.scatter_peer_ok[gi] = can_scatter;
    if(!can_collect) info.need_host_stage_collect = true;
    if(!can_scatter) info.need_host_stage_scatter = true;
  }
}

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
    if(!can) info.need_host_stage = true;
  }
}

static void broadcast_segment_to_gpus_async(const OrbitGenCtx& gen,
                                            int gen_slot,
                                            const BaselineBroadcastInfo& info,
                                            std::vector<GpuCtx>& ctx,
                                            bool have_host_xyz)
{
  const OrbitSegSlot& src = gen.slots[gen_slot];
  #pragma omp parallel for num_threads((int)ctx.size())
  for(int gi=0; gi<(int)ctx.size(); ++gi){
    CHECK_CUDA(cudaSetDevice(ctx[gi].dev));
    cudaStream_t stream = ctx[gi].xfer_stream;
    GpuSegSlot &dst = ctx[gi].slots[gen_slot];
    dst.segN = src.segN;
    dst.segN_half = src.tlen * UNIQUE_BASELINES_PER_T;
    dst.tlen = src.tlen;

    CHECK_CUDA(cudaStreamWaitEvent(stream, src.ready, 0));
    auto copy_arr = [&](float* dst_ptr, const float* src_dev, const float* src_host, size_t count){
      size_t bytes = count * sizeof(float);
      if(ctx[gi].dev == gen.dev){
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_dev, bytes, cudaMemcpyDeviceToDevice, stream));
      }else if(info.peer_ok[gi]){
        CHECK_CUDA(cudaMemcpyPeerAsync(dst_ptr, ctx[gi].dev, src_dev, gen.dev, bytes, stream));
      }else{
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_host, bytes, cudaMemcpyHostToDevice, stream));
      }
    };

    copy_arr(dst.d_u,  src.d_u,  src.h_u,  (size_t)dst.segN_half);
    copy_arr(dst.d_v,  src.d_v,  src.h_v,  (size_t)dst.segN_half);
    copy_arr(dst.d_w,  src.d_w,  src.h_w,  (size_t)dst.segN_half);
    copy_arr(dst.d_x1, src.d_x1, src.h_x1, (size_t)dst.segN_half);
    copy_arr(dst.d_y1, src.d_y1, src.h_y1, (size_t)dst.segN_half);
    copy_arr(dst.d_z1, src.d_z1, src.h_z1, (size_t)dst.segN_half);
    copy_arr(dst.d_x2, src.d_x2, src.h_x2, (size_t)dst.segN_half);
    copy_arr(dst.d_y2, src.d_y2, src.h_y2, (size_t)dst.segN_half);
    copy_arr(dst.d_z2, src.d_z2, src.h_z2, (size_t)dst.segN_half);

    int b2 = 256;
    int g2 = (dst.segN_half + b2 - 1) / b2;
    normalize3_inplace_kernel<<<g2,b2,0,stream>>>(dst.d_x1, dst.d_y1, dst.d_z1, dst.segN_half);
    normalize3_inplace_kernel<<<g2,b2,0,stream>>>(dst.d_x2, dst.d_y2, dst.d_z2, dst.segN_half);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(dst.ready, stream));
  }
}





// -------- direct 3D recon (half-sym + tilecone) --------


struct GpuDirectExtra {
  float* d_dcf = nullptr;
  float* d_pairw_half[2] = {nullptr, nullptr};
  float2* d_reduce_parts[2] = {nullptr, nullptr};
};

__device__ __forceinline__ int half_to_full_pos_idx_dev(int ih){
  int group = ih / UNIQUE_BASELINES_PER_T;
  int j = ih - group * UNIQUE_BASELINES_PER_T;
  return group * SIGNED_BASELINES_PER_T + j;
}


__global__ void reduce_phase_halfsym_fused_kernel(
    const float2* __restrict__ reduce_parts,
    int parts_stride,
    int parts_count,
    const float* __restrict__ w_half,
    int N_half,
    float2* __restrict__ Viss_half_out)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= N_half) return;

  float re = 0.0f;
  float im = 0.0f;
  for(int p=0; p<parts_count; ++p){
    float2 z = reduce_parts[(size_t)p * (size_t)parts_stride + (size_t)ih];
    re += z.x;
    im += z.y;
  }

  float ang = -2.0f * (float)M_PI * w_half[ih];
  float s, c;
  sincos_fast(ang, &s, &c);
  Viss_half_out[ih] = make_float2(re*c - im*s, re*s + im*c);
}

__global__ void compute_pair_weight_half_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    int N_half,
    const float* __restrict__ dcf,
    int dcf_len,
    float* __restrict__ pair_weight_half)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= N_half) return;

  float uu = u[ih], vv = v[ih], ww = w[ih];
  float bl = sqrtf(uu*uu + vv*vv + ww*ww);

  float wgt = 0.0f;
  if(dcf_len > 0){
    if(!isfinite(bl) || bl <= 1e-20f){
      wgt = dcf[0];
    } else {
      int gs = (int)ceilf((bl - 0.25f) / 0.5f) + 1;
      if(gs < 0) gs = 0;
      if(gs >= dcf_len) gs = dcf_len - 1;

      float s = ww / bl;
      s = fminf(1.0f, fmaxf(-1.0f, s));
      float c = sqrtf(fmaxf(0.0f, 1.0f - s*s));
      float a = 1.0f - 4.0f * s * s;
      float mag_sqrt = sqrtf(fabsf(a));
      float gdg = 0.0f;
      if(c > 1e-12f) gdg = 1.5f * mag_sqrt / c;
      wgt = dcf[gs] * gdg;
    }
  }

  if(!isfinite(wgt) || wgt < 0.0f) wgt = 0.0f;
  if(wgt > 0.125f) wgt = 0.125f;
  pair_weight_half[ih] = 2.0f * wgt;
}

// Apply the half-symmetry / DCF pair weight once per visibility sample.
// This removes pair-weight loads, multiply, and finite checks from the hot
// per-pixel recon loops. Non-finite visibility samples are sanitized here,
// preserving the previous "skip bad z" behavior at much lower frequency.
__global__ void apply_pair_weight_to_viss_half_kernel(
    float2* __restrict__ Viss_half,
    const float* __restrict__ pair_weight_half,
    int N_half)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  if(ih >= N_half) return;

  float pw = pair_weight_half[ih];
  float2 z = Viss_half[ih];
  if(!(pw > 0.0f) || !isfinite(pw) || !isfinite(z.x) || !isfinite(z.y)){
    Viss_half[ih] = make_float2(0.0f, 0.0f);
  } else {
    Viss_half[ih] = make_float2(z.x * pw, z.y * pw);
  }
}


template<int TILE_BL>
__global__ void build_recon_task_flags_chunk_kernel(
    int ntile,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    int b0,
    int chunk_n,
    float cosphi,
    unsigned char* __restrict__ vv_flags,
    unsigned char* __restrict__ mixed_flags)
{
  int nblocks = (chunk_n + TILE_BL - 1) / TILE_BL;
  int task_count = ntile * nblocks;
  int task = blockIdx.x * blockDim.x + threadIdx.x;
  if(task >= task_count) return;

  int blk = task / ntile;
  int tile = task - blk * ntile;
  int base = b0 + blk * TILE_BL;
  int tileN = min(TILE_BL, chunk_n - blk * TILE_BL);

  float cx = tile_cx[tile], cy = tile_cy[tile], cz = tile_cz[tile];
  float cosA = tile_cosA[tile], sinA = tile_sinA[tile];

  bool all_visible_all = true;
  bool all_hidden_all = true;
  for(int t=0; t<tileN; ++t){
    int ih = base + t;
    bool allv=false, allh=false;
    operator_classify_pair_tile_dirs(x1[ih], y1[ih], z1[ih], x2[ih], y2[ih], z2[ih], cx, cy, cz, cosA, sinA, cosphi, allv, allh);
    if(!allv) all_visible_all = false;
    if(!allh) all_hidden_all = false;
    if(!all_visible_all && !all_hidden_all) break;
  }

  vv_flags[task] = all_visible_all ? (unsigned char)1 : (unsigned char)0;
  mixed_flags[task] = (!all_visible_all && !all_hidden_all) ? (unsigned char)1 : (unsigned char)0;
}

template<int TILE_PIX, int TILE_BL>
__global__ void recon_3d_direct_tasklist_vv_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float2* __restrict__ Viss_half,
    int b0,
    int chunk_n,
    const int* __restrict__ task_ids,
    int num_tasks,
    float* __restrict__ Cacc)
{
  int task_idx = blockIdx.x;
  if(task_idx >= num_tasks) return;
  int task = task_ids[task_idx];
  int blk = task / ntile;
  int tile = task - blk * ntile;
  int base = b0 + blk * TILE_BL;
  int tileBL = min(TILE_BL, chunk_n - blk * TILE_BL);

  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float2 sV[TILE_BL];

  int lane = threadIdx.x;
  long long pix = (long long)tile * TILE_PIX + lane;
  bool active = (pix < n_chunk);

  float lp=0.0f, mp=0.0f, npv=0.0f;
  if(active){ lp=l[pix]; mp=m[pix]; npv=n[pix]; }

  for(int t=lane; t<tileBL; t+=blockDim.x){
    int ih = base + t;
    su[t] = u[ih]; sv[t] = v[ih]; sw[t] = w[ih];
    sV[t] = Viss_half[ih];
  }
  __syncthreads();

  float acc_re = 0.0f;
  if(active){
    #pragma unroll 2
    for(int t=0; t<tileBL; ++t){
      float2 z = sV[t];
#if OP_RECON_SKIP_ZERO_WEIGHTED_VISS
      if(z.x == 0.0f && z.y == 0.0f) continue;
#endif
#if OP_RECON_DEBUG_CHECK_FINITE
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
#endif
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      acc_re += z.x * c - z.y * s;
    }
  }
  if(active) Cacc[pix] += acc_re;
}

template<int TILE_PIX, int TILE_BL>
__global__ void recon_3d_direct_tasklist_mixed_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float2* __restrict__ Viss_half,
    int b0,
    int chunk_n,
    float cosphi,
    const int* __restrict__ task_ids,
    int num_tasks,
    float* __restrict__ Cacc)
{
  int task_idx = blockIdx.x;
  if(task_idx >= num_tasks) return;
  int task = task_ids[task_idx];
  int blk = task / ntile;
  int tile = task - blk * ntile;
  int base = b0 + blk * TILE_BL;
  int tileBL = min(TILE_BL, chunk_n - blk * TILE_BL);

  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float sdx1[TILE_BL], sdy1[TILE_BL], sdz1[TILE_BL];
  __shared__ float sdx2[TILE_BL], sdy2[TILE_BL], sdz2[TILE_BL];
  __shared__ float2 sV[TILE_BL];
  __shared__ unsigned short svv_idx[TILE_BL], smix_idx[TILE_BL];
  __shared__ int nvv, nmix;

  int lane = threadIdx.x;
  long long pix = (long long)tile * TILE_PIX + lane;
  bool active = (pix < n_chunk);
  float lp=0.0f, mp=0.0f, npv=0.0f;
  if(active){ lp=l[pix]; mp=m[pix]; npv=n[pix]; }

  for(int t=lane; t<tileBL; t+=blockDim.x){
    int ih = base + t;
    su[t]=u[ih]; sv[t]=v[ih]; sw[t]=w[ih];
    sdx1[t]=x1[ih]; sdy1[t]=y1[ih]; sdz1[t]=z1[ih];
    sdx2[t]=x2[ih]; sdy2[t]=y2[ih]; sdz2[t]=z2[ih];
    sV[t]=Viss_half[ih];
  }
  if(lane==0){ nvv=0; nmix=0; }
  __syncthreads();

  float tcx = tile_cx[tile], tcy = tile_cy[tile], tcz = tile_cz[tile];
  float tcosA = tile_cosA[tile], tsinA = tile_sinA[tile];
  if(lane < tileBL){
    bool allv=false, allh=false;
    operator_classify_pair_tile_dirs(sdx1[lane],sdy1[lane],sdz1[lane],sdx2[lane],sdy2[lane],sdz2[lane],tcx,tcy,tcz,tcosA,tsinA,cosphi,allv,allh);
    if(allv){
      int dst = atomicAdd(&nvv, 1);
      svv_idx[dst] = (unsigned short)lane;
    } else if(!allh){
      int dst = atomicAdd(&nmix, 1);
      smix_idx[dst] = (unsigned short)lane;
    }
  }
  __syncthreads();

  float acc_re = 0.0f;
  if(active){
    #pragma unroll 2
    for(int ii=0; ii<nvv; ++ii){
      int t = (int)svv_idx[ii];
      float2 z = sV[t];
#if OP_RECON_SKIP_ZERO_WEIGHTED_VISS
      if(z.x == 0.0f && z.y == 0.0f) continue;
#endif
#if OP_RECON_DEBUG_CHECK_FINITE
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
#endif
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      acc_re += z.x * c - z.y * s;
    }
    #pragma unroll 2
    for(int ii=0; ii<nmix; ++ii){
      int t = (int)smix_idx[ii];
      float2 z = sV[t];
#if OP_RECON_SKIP_ZERO_WEIGHTED_VISS
      if(z.x == 0.0f && z.y == 0.0f) continue;
#endif
#if OP_RECON_DEBUG_CHECK_FINITE
      if(!isfinite(z.x) || !isfinite(z.y)) continue;
#endif
      if(!operator_point_visible_pair_dirs(sdx1[t],sdy1[t],sdz1[t],sdx2[t],sdy2[t],sdz2[t],lp,mp,npv,cosphi)) continue;
      float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
      float s, c; sincos_fast(phase, &s, &c);
      acc_re += z.x * c - z.y * s;
    }
  }
  if(active) Cacc[pix] += acc_re;
}

template<int TILE_BL>
static inline void enqueue_recon_plan_build(GpuCtx& gctx,
                                            const GpuSegSlot& slot,
                                            OperatorPlanBuffer& plan,
                                            int b0,
                                            int chunk_n,
                                            float cosphi)
{
  cudaStream_t stream = gctx.reduce_stream;
  CHECK_CUDA(cudaStreamWaitEvent(stream, plan.compute_done, 0));
  int nblocks = (chunk_n + TILE_BL - 1) / TILE_BL;
  int task_count = gctx.ntile_recon * nblocks;
  plan.last_b0 = b0;
  plan.last_chunk_n = chunk_n;
  int grid = (task_count + 255) / 256;
  build_recon_task_flags_chunk_kernel<TILE_BL><<<grid,256,0,stream>>>(
      gctx.ntile_recon,
      gctx.d_tile_cx_recon, gctx.d_tile_cy_recon, gctx.d_tile_cz_recon,
      gctx.d_tile_cosA_recon, gctx.d_tile_sinA_recon,
      slot.d_x1, slot.d_y1, slot.d_z1,
      slot.d_x2, slot.d_y2, slot.d_z2,
      b0, chunk_n, cosphi,
      plan.d_vv_flags, plan.d_mixed_flags);
  CHECK_CUDA(cudaPeekAtLastError());

  cub::CountingInputIterator<int> counting(0);
  CHECK_CUDA(cub::DeviceSelect::Flagged(plan.d_select_temp_vv, plan.select_temp_vv_bytes,
      counting, plan.d_vv_flags, plan.d_vv_tasks, plan.d_num_vv, task_count, stream));
  CHECK_CUDA(cub::DeviceSelect::Flagged(plan.d_select_temp_mixed, plan.select_temp_mixed_bytes,
      counting, plan.d_mixed_flags, plan.d_mixed_tasks, plan.d_num_mixed, task_count, stream));
  CHECK_CUDA(cudaMemcpyAsync(plan.h_num_vv, plan.d_num_vv, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(plan.h_num_mixed, plan.d_num_mixed, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaEventRecord(plan.ready, stream));
}

static inline ReconGpuLoadStats run_recon_planaware_segment(GpuCtx& gctx,
                                               GpuSegSlot& slot,
                                               int segN_half,
                                               float cosphi)
{
  ReconGpuLoadStats stats;
  if(segN_half <= 0) return stats;
  stats.gpu_dev = gctx.dev;
  const int CH = OP_RECON_PLAN_BL_CHUNK;
  const int total_chunks = (segN_half + CH - 1) / CH;

  int primed = min(2, total_chunks);
  for(int pi=0; pi<primed; ++pi){
    int b0 = pi * CH;
    int cn = min(CH, segN_half - b0);
    enqueue_recon_plan_build<OP_RECON_TASK_BL>(gctx, slot, gctx.recon_plan[pi], b0, cn, cosphi);
  }

  for(int ci=0; ci<total_chunks; ++ci){
    int cur = ci & 1;
    OperatorPlanBuffer& plan = gctx.recon_plan[cur];
    CHECK_CUDA(cudaEventSynchronize(plan.ready));

    int num_vv = *plan.h_num_vv;
    int num_mixed = *plan.h_num_mixed;
    int b0 = plan.last_b0;
    int cn = plan.last_chunk_n;
    stats.chunks += 1;
    stats.vv_tasks += (long long)num_vv;
    stats.mixed_tasks += (long long)num_mixed;
    if(num_vv == 0 && num_mixed == 0) stats.zero_task_chunks += 1;
    else if(num_mixed == 0) stats.vv_only_chunks += 1;
    else stats.mixed_chunks += 1;
    cudaStream_t cstream = gctx.compute_stream;
    CHECK_CUDA(cudaStreamWaitEvent(cstream, slot.pairw_ready, 0));
    if(num_vv > 0){
      recon_3d_direct_tasklist_vv_real<OP_RECON_TASK_TILE_PIX, OP_RECON_TASK_BL><<<num_vv, OP_RECON_TASK_TILE_PIX, 0, cstream>>>(
          gctx.recon_n_chunk, gctx.d_l_recon, gctx.d_m_recon, gctx.d_n_recon, gctx.ntile_recon,
          slot.d_u, slot.d_v, slot.d_w,
          slot.d_Viss,
          b0, cn, plan.d_vv_tasks, num_vv, gctx.d_Cacc);
      CHECK_CUDA(cudaPeekAtLastError());
    }
    if(num_mixed > 0){
      recon_3d_direct_tasklist_mixed_real<OP_RECON_TASK_TILE_PIX, OP_RECON_TASK_BL><<<num_mixed, OP_RECON_TASK_TILE_PIX, 0, cstream>>>(
          gctx.recon_n_chunk, gctx.d_l_recon, gctx.d_m_recon, gctx.d_n_recon,
          gctx.d_tile_cx_recon, gctx.d_tile_cy_recon, gctx.d_tile_cz_recon,
          gctx.d_tile_cosA_recon, gctx.d_tile_sinA_recon, gctx.ntile_recon,
          slot.d_u, slot.d_v, slot.d_w,
          slot.d_x1, slot.d_y1, slot.d_z1, slot.d_x2, slot.d_y2, slot.d_z2,
          slot.d_Viss,
          b0, cn, cosphi, plan.d_mixed_tasks, num_mixed, gctx.d_Cacc);
      CHECK_CUDA(cudaPeekAtLastError());
    }
    CHECK_CUDA(cudaEventRecord(plan.compute_done, cstream));

    int ni = ci + 2;
    if(ni < total_chunks){
      int b0n = ni * CH;
      int cnn = min(CH, segN_half - b0n);
      enqueue_recon_plan_build<OP_RECON_TASK_BL>(gctx, slot, gctx.recon_plan[cur], b0n, cnn, cosphi);
    }

  }
  return stats;
}

template<int TILE_PIX, int TILE_BL, bool DO_BLOCKAGE>
__global__ void recon_3d_direct_halfsym_tilecone_real(
    long long n_chunk,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    const float* __restrict__ tile_cx,
    const float* __restrict__ tile_cy,
    const float* __restrict__ tile_cz,
    const float* __restrict__ tile_cosA,
    const float* __restrict__ tile_sinA,
    int ntile,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ x1,
    const float* __restrict__ y1,
    const float* __restrict__ z1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float2* __restrict__ Viss_half,
    int N_half,
    float cosphi,
    float* __restrict__ Cacc)
{
  __shared__ float su[TILE_BL], sv[TILE_BL], sw[TILE_BL];
  __shared__ float sdx1[TILE_BL], sdy1[TILE_BL], sdz1[TILE_BL];
  __shared__ float sdx2[TILE_BL], sdy2[TILE_BL], sdz2[TILE_BL];
  __shared__ float2 sV[TILE_BL];
  __shared__ unsigned char sallv1[TILE_BL], sallh1[TILE_BL], sallv2[TILE_BL], sallh2[TILE_BL];

  int tile = blockIdx.x;
  if(tile >= ntile) return;

  int lane = threadIdx.x;
  long long pix = (long long)tile * TILE_PIX + lane;
  bool active = (pix < n_chunk);

  float lp = 0.0f, mp = 0.0f, npv = 0.0f;
  if(active){
    lp = l[pix];
    mp = m[pix];
    npv = n[pix];
  }

  float tcx = 0.0f, tcy = 0.0f, tcz = 0.0f, tcosA = 1.0f, tsinA = 0.0f;
  if constexpr (DO_BLOCKAGE){
    if(tile < ntile){
      tcx = tile_cx[tile];
      tcy = tile_cy[tile];
      tcz = tile_cz[tile];
      tcosA = tile_cosA[tile];
      tsinA = tile_sinA[tile];
    }
  }

  float acc_re = 0.0f;

  for(int b0 = 0; b0 < N_half; b0 += TILE_BL){
    int tileN = min(TILE_BL, N_half - b0);

    for(int t = lane; t < tileN; t += blockDim.x){
      int ih = b0 + t;
      su[t]   = u[ih];
      sv[t]   = v[ih];
      sw[t]   = w[ih];
      sdx1[t] = x1[ih];  sdy1[t] = y1[ih];  sdz1[t] = z1[ih];
      sdx2[t] = x2[ih];  sdy2[t] = y2[ih];  sdz2[t] = z2[ih];
      sV[t]   = Viss_half[ih];
    }
    __syncthreads();

    if constexpr (DO_BLOCKAGE){
      if(lane < tileN){
        bool allv=false, allh=false;
        operator_classify_pair_tile_dirs(sdx1[lane],sdy1[lane],sdz1[lane],sdx2[lane],sdy2[lane],sdz2[lane],tcx,tcy,tcz,tcosA,tsinA,cosphi,allv,allh);
        sallv1[lane] = (unsigned char)(allv ? 1 : 0);
        sallh1[lane] = (unsigned char)(allh ? 1 : 0);
        sallv2[lane] = 0;
        sallh2[lane] = 0;
      }
    }
    __syncthreads();

    if(active){
      #pragma unroll 2
      for(int t = 0; t < tileN; ++t){
        if constexpr (DO_BLOCKAGE){
          if(sallh1[t]) continue;
          if(!sallv1[t] && !operator_point_visible_pair_dirs(sdx1[t],sdy1[t],sdz1[t],sdx2[t],sdy2[t],sdz2[t],lp,mp,npv,cosphi)) continue;
        }

        float phase = operator_adjoint_phase(su[t], sv[t], sw[t], lp, mp, npv);
        float s, c;
        sincos_fast(phase, &s, &c);

        float2 z = sV[t];
        acc_re += z.x * c - z.y * s;
      }
    }
    __syncthreads();
  }

  if(active) Cacc[pix] += (float)acc_re;
}
