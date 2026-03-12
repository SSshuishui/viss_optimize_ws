#pragma once

#include "common.hpp"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_reduce.cuh>

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

