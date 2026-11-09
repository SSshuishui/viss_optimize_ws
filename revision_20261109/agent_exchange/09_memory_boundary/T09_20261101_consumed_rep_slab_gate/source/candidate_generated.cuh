// Independently derived from frozen T09 bodies; MODE4/5 fix FP32 microchunk grouping.
// MODE0 original-header bitwise gate remains mandatory; no tested-repair claim.
#pragma once
__device__ __forceinline__ void add_kahan64(double x,double& s,double& c){
 double y=__dsub_rn(x,c),t=__dadd_rn(s,y);c=__dsub_rn(__dsub_rn(t,s),y);s=t;
}
__global__ void pix2lmn_nest_selected(
    int nside,
    const unsigned int* ids,
    int chunkN,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= chunkN) return;

  unsigned int ipix = ids[tid];
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

template<int TILE_PIX, bool DO_BLOCKAGE, int MODE>
__global__ void slab_candidate(
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
    const float* __restrict__ invn1,
    const float* __restrict__ x2,
    const float* __restrict__ y2,
    const float* __restrict__ z2,
    const float* __restrict__ invn2,
    int N_half,
    float cosphi,
    float2* __restrict__ Vpart, double2* __restrict__ Dstate, double2* __restrict__ Dcomp)
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
  if constexpr(MODE==1){if(active){acc_re=Vpart[ih].x;acc_im=Vpart[ih].y;}}
  double ar=0,ai=0,cr=0,ci=0;
  if constexpr(MODE>=2){if(active){ar=Dstate[ih].x;ai=Dstate[ih].y;cr=Dcomp[ih].x;ci=Dcomp[ih].y;}}
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
  float* sN = sM + TILE_PIX;
  float* sNM1 = sN + TILE_PIX;

  for(int tid=0; tid<ntile; ++tid){
    long long p0 = (long long)tid * TILE_PIX;
    int tileN = (int)min((long long)TILE_PIX, n_chunk - p0);
    for(int lane = threadIdx.x; lane < tileN; lane += blockDim.x){
      long long p = p0 + lane;
      sB[lane] = B[p];
      sL[lane] = l[p];
      sM[lane] = m[p];
      sN[lane] = n[p];
      sNM1[lane] = nm1[p];
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
      if(!DO_BLOCKAGE || all_visible){
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], nm1v = sNM1[k0];
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          if constexpr(MODE<2 || MODE>=4){acc_re += bp*c;acc_im += bp*s;}
          else if constexpr(MODE==2){
            add_kahan64((double)__fmul_rn(bp,c),ar,cr);
            add_kahan64((double)__fmul_rn(bp,s),ai,ci);
          }else{
            const double kd=-2.0*3.141592653589793238462643383279502884;
            double ph=fma(kd*(double)w0,(double)nm1v,fma(kd*(double)v0,(double)mp,(kd*(double)u0)*(double)lp));
            double sd,cd;sincos(ph,&sd,&cd);
            add_kahan64((double)bp*cd,ar,cr);add_kahan64((double)bp*sd,ai,ci);
          }
        }
      }else{
        #pragma unroll 4
        for(int k0=0; k0<tileN; ++k0){
          float lp = sL[k0], mp = sM[k0], npv = sN[k0], nm1v = sNM1[k0];
          float c1=(lp*x1i + mp*y1i + npv*z1i)*in1;
          float c2=(lp*x2i + mp*y2i + npv*z2i)*in2;
          if(c1<cosphi || c2<cosphi) continue;
          float ang = fmaf(kw, nm1v, fmaf(kv, mp, ku * lp));
          float s,c; sincos_fast(ang,&s,&c);
          float bp=sB[k0];
          if constexpr(MODE<2 || MODE>=4){acc_re += bp*c;acc_im += bp*s;}
          else if constexpr(MODE==2){
            add_kahan64((double)__fmul_rn(bp,c),ar,cr);
            add_kahan64((double)__fmul_rn(bp,s),ai,ci);
          }else{
            const double kd=-2.0*3.141592653589793238462643383279502884;
            double ph=fma(kd*(double)w0,(double)nm1v,fma(kd*(double)v0,(double)mp,(kd*(double)u0)*(double)lp));
            double sd,cd;sincos(ph,&sd,&cd);
            add_kahan64((double)bp*cd,ar,cr);add_kahan64((double)bp*sd,ai,ci);
          }
        }
      }
    }
    __syncthreads();
    if constexpr(MODE>=4){
      constexpr int MICRO_TILES=(MODE==4)?128:256;
      if(active && (((tid+1)%MICRO_TILES)==0 || tid+1==ntile)){
        ar=__dadd_rn(ar,(double)acc_re);ai=__dadd_rn(ai,(double)acc_im);
        acc_re=0.0f;acc_im=0.0f;
      }
    }
  }

  if(active){
    if constexpr(MODE<2) Vpart[ih]=make_float2(acc_re,acc_im);
    else {Dstate[ih]=make_double2(ar,ai);Dcomp[ih]=make_double2(cr,ci);}
  }
}
