#pragma once

#include "common_mirror_tilecone.hpp"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

__device__ __forceinline__ float2 cadd(float2 a,float2 b){ return make_float2(a.x+b.x,a.y+b.y); }
__device__ __forceinline__ float2 cmul(float2 a,float2 b){ return make_float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x); }
__device__ __forceinline__ float2 cexpj(float phase){ float s,c; __sincosf(phase,&s,&c); return make_float2(c,s); }
__device__ __forceinline__ int round_away_from_zero(float x){ return (x>=0.0f)? (int)floorf(x+0.5f) : (int)ceilf(x-0.5f); }
__device__ __forceinline__ int halfidx_to_fullpos_dev(int ih){
  int group = ih / UNIQUE_BASELINES_PER_T;
  int j = ih - group * UNIQUE_BASELINES_PER_T;
  return group * SIGNED_BASELINES_PER_T + j;
}


static constexpr int VISS_TILE_PIX = 256;

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
  long long p0 = (long long)tid * VISS_TILE_PIX;
  int tileN = (int)min((long long)VISS_TILE_PIX, n_chunk - p0);
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

__device__ __forceinline__ void classify_tile_cone(float dotc, float cosA, float sinA, float cosphi, bool &all_vis, bool &all_hid){
  dotc = fminf(1.0f, fmaxf(-1.0f, dotc));
  float sind = sqrtf(fmaxf(0.0f, 1.0f - dotc*dotc));
  float lower = dotc * cosA - sind * sinA;
  float upper = dotc * cosA + sind * sinA;
  all_vis = (lower >= cosphi);
  all_hid = (upper <  cosphi);
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

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_tilecone(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
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
  float ux1=0.0f, uy1=0.0f, uz1=0.0f, ux2=0.0f, uy2=0.0f, uz2=0.0f;

  if(active){
    int group = ih / UNIQUE_BASELINES_PER_T;
    int j     = ih - group * UNIQUE_BASELINES_PER_T;
    i         = group * SIGNED_BASELINES_PER_T + j;
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    if constexpr (DO_BLOCKAGE){
      x1i = x1[i]; y1i = y1[i]; z1i = z1[i]; in1 = invn1[i];
      x2i = x2[i]; y2i = y2[i]; z2i = z2[i]; in2 = invn2[i];
      ux1 = x1i * in1; uy1 = y1i * in1; uz1 = z1i * in1;
      ux2 = x2i * in2; uy2 = y2i * in2; uz2 = z2i * in2;
    }
  }

  float acc_re = 0.0f, acc_im = 0.0f;
  const float k = -2.0f * (float)M_PI;

  extern __shared__ float smem[];
  float* sB = smem;
  float* sL = sB + TILE_PIX;
  float* sM = sL + TILE_PIX;
  float* sN = sM + TILE_PIX;

  for(int tid=0; tid<ntile; ++tid){
    long long p0 = (long long)tid * TILE_PIX;
    int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
    for(int lane = threadIdx.x; lane < tileN; lane += blockDim.x){
      long long p = p0 + lane;
      sB[lane] = B[p];
      sL[lane] = l[p];
      sM[lane] = m[p];
      sN[lane] = n[p];
    }
    __syncthreads();

    bool all_visible = false;
    bool all_hidden  = false;
    if(active && DO_BLOCKAGE){
      float cx = tile_cx[tid], cy = tile_cy[tid], cz = tile_cz[tid];
      float cosA = tile_cosA[tid], sinA = tile_sinA[tid];
      bool vis1=false, hid1=false, vis2=false, hid2=false;
      classify_tile_cone(ux1*cx + uy1*cy + uz1*cz, cosA, sinA, cosphi, vis1, hid1);
      classify_tile_cone(ux2*cx + uy2*cy + uz2*cz, cosA, sinA, cosphi, vis2, hid2);
      all_hidden = hid1 || hid2;
      all_visible = vis1 && vis2;
    }

    if(active && !(DO_BLOCKAGE && all_hidden)) {
      #pragma unroll 4
      for(int k0=0; k0<tileN; ++k0){
        float lp = sL[k0], mp = sM[k0], npv = sN[k0];
        if constexpr (DO_BLOCKAGE){
          if(!all_visible){
            float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
            float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
            if(c1<cosphi || c2<cosphi) continue;
          }
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

template<int TILE_PIX, bool DO_BLOCKAGE>
__global__ void viss_partial_all_halfsym_reuse_sat(
    const float* __restrict__ B,
    const float* __restrict__ l,
    const float* __restrict__ m,
    const float* __restrict__ n,
    long long n_chunk,
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ satx,
    const float* __restrict__ saty,
    const float* __restrict__ satz,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart)
{
  int ih = blockIdx.x * blockDim.x + threadIdx.x;
  bool active = (ih < N_half);

  int group = 0, j = 0, i = 0;
  float u0 = 0.0f, v0 = 0.0f, w0 = 0.0f;
  int sat_m = 0, sat_n = 0;
  if(active){
    group = ih / UNIQUE_BASELINES_PER_T;
    j     = ih - group * UNIQUE_BASELINES_PER_T;
    i     = group * SIGNED_BASELINES_PER_T + j;
    u0 = u[i]; v0 = v[i]; w0 = w[i];
    sat_m = d_pair_m[j];
    sat_n = d_pair_n[j];
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
      int sat_base = group * SATNUM;
      #pragma unroll 4
      for(int k0=0; k0<tileN; ++k0){
        float lp = sL[k0], mp = sM[k0], npv = sN[k0];
        if constexpr (DO_BLOCKAGE){
          bool vis[SATNUM];
          #pragma unroll
          for(int s=0; s<SATNUM; ++s){
            float x = satx[sat_base + s];
            float y = saty[sat_base + s];
            float z = satz[sat_base + s];
            float c = (lp*x + mp*y + npv*z) * rsqrtf(x*x + y*y + z*z);
            vis[s] = (c >= cosphi);
          }
          if(!(vis[sat_m] && vis[sat_n])) continue;
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


struct GpuSegSlot {
  float* d_pos=nullptr;
  float* d_satx=nullptr;
  float* d_saty=nullptr;
  float* d_satz=nullptr;

  float* d_u=nullptr; float* d_v=nullptr; float* d_w=nullptr;
  float* d_x1=nullptr; float* d_y1=nullptr; float* d_z1=nullptr;
  float* d_x2=nullptr; float* d_y2=nullptr; float* d_z2=nullptr;
  float* d_invn1=nullptr; float* d_invn2=nullptr;

  float2* d_Vpart=nullptr;
  float2* d_Viss=nullptr;
  float2* h_Vpart=nullptr;

  cudaEvent_t ready=nullptr;
  int segN=0;
  int segN_half=0;
  int tlen=0;
};

struct GpuCtx {
  int dev=0;
  cudaStream_t compute_stream=nullptr;
  cudaStream_t xfer_stream=nullptr;
  long long pix0=0,pix1=0,n_chunk=0;

  float* d_B=nullptr;
  float* d_l=nullptr;
  float* d_m=nullptr;
  float* d_n=nullptr;
  int ntile=0;
  float* d_tile_cx=nullptr;
  float* d_tile_cy=nullptr;
  float* d_tile_cz=nullptr;
  float* d_tile_cosA=nullptr;
  float* d_tile_sinA=nullptr;

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
  #pragma omp parallel for num_threads(64)
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
      }else{
        CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_host, bytes, cudaMemcpyHostToDevice, stream));
      }
    };

    copy_arr(dst.d_pos, src.d_pos, src.h_pos, (size_t)src.tlen * (3*SATNUM));
    copy_arr(dst.d_u,  src.d_u,  src.h_u,  (size_t)src.segN);
    copy_arr(dst.d_v,  src.d_v,  src.h_v,  (size_t)src.segN);
    copy_arr(dst.d_w,  src.d_w,  src.h_w,  (size_t)src.segN);
    copy_arr(dst.d_x1, src.d_x1, src.h_x1, (size_t)src.segN);
    copy_arr(dst.d_y1, src.d_y1, src.h_y1, (size_t)src.segN);
    copy_arr(dst.d_z1, src.d_z1, src.h_z1, (size_t)src.segN);
    copy_arr(dst.d_x2, src.d_x2, src.h_x2, (size_t)src.segN);
    copy_arr(dst.d_y2, src.d_y2, src.h_y2, (size_t)src.segN);
    copy_arr(dst.d_z2, src.d_z2, src.h_z2, (size_t)src.segN);

    int b2 = 256;
    int g_pos = (dst.tlen * SATNUM + b2 - 1) / b2;
    build_sat_raw_from_pos<<<g_pos,b2,0,stream>>>(dst.d_pos, dst.d_satx, dst.d_saty, dst.d_satz, dst.tlen);

    int g2 = (dst.segN + b2 - 1) / b2;
    invnorm3_kernel<<<g2,b2,0,stream>>>(dst.d_x1, dst.d_y1, dst.d_z1, dst.d_invn1, dst.segN);
    invnorm3_kernel<<<g2,b2,0,stream>>>(dst.d_x2, dst.d_y2, dst.d_z2, dst.d_invn2, dst.segN);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaEventRecord(dst.ready, stream));
  }
}



