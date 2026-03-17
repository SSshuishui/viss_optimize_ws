#include "common_mirror.hpp"

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

void orbit_gen_init(OrbitGenCtx& gen, int dev, float lambda_m, int dayid, int max_tlen, uint64_t seed){
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

  for(int slot=0; slot<2; ++slot){
    OrbitSegSlot &sg = gen.slots[slot];
    CHECK_CUDA(cudaMalloc(&sg.d_pos, (size_t)max_tlen * (3*SATNUM) * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_u,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_v,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_w,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_x1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_y1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_z1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_x2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_y2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sg.d_z2, (size_t)gen.max_segN * sizeof(float)));

    CHECK_CUDA(cudaMallocHost(&sg.h_pos, (size_t)max_tlen * (3*SATNUM) * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_u,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_v,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_w,  (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_x1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_y1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_z1, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_x2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_y2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&sg.h_z2, (size_t)gen.max_segN * sizeof(float)));
    CHECK_CUDA(cudaEventCreateWithFlags(&sg.ready, cudaEventDisableTiming));
  }

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

void orbit_gen_make_segment_async(OrbitGenCtx& gen, int slot_idx, int t0, int tlen, bool copy_all_host){
  CHECK_CUDA(cudaSetDevice(gen.dev));
  OrbitSegSlot &sg = gen.slots[slot_idx];
  sg.t0 = t0;
  sg.tlen = tlen;
  sg.segN = tlen * SIGNED_BASELINES_PER_T;

  const int BS = 256;
  int GS = (tlen + BS - 1) / BS;
  k_gather_pos_range<<<GS, BS, 0, gen.stream>>>(gen.d_x, gen.d_y, gen.d_z, sg.d_pos, gen.d_t1s,
                                                gen.OrbitRes, gen.ProcessionCount, gen.segLen, t0, tlen);
  CHECK_CUDA(cudaPeekAtLastError());

  GS = (sg.segN + BS - 1) / BS;
  k_compute_uvw_xyz<<<GS, BS, 0, gen.stream>>>(sg.d_pos, tlen, gen.lambda_m,
                                               sg.d_u, sg.d_v, sg.d_w,
                                               sg.d_x1, sg.d_y1, sg.d_z1,
                                               sg.d_x2, sg.d_y2, sg.d_z2);
  CHECK_CUDA(cudaPeekAtLastError());

  CHECK_CUDA(cudaMemcpyAsync(sg.h_pos, sg.d_pos, (size_t)sg.tlen*(3*SATNUM)*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(sg.h_u, sg.d_u, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(sg.h_v, sg.d_v, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  CHECK_CUDA(cudaMemcpyAsync(sg.h_w, sg.d_w, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  if(copy_all_host){
    CHECK_CUDA(cudaMemcpyAsync(sg.h_x1, sg.d_x1, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(sg.h_y1, sg.d_y1, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(sg.h_z1, sg.d_z1, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(sg.h_x2, sg.d_x2, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(sg.h_y2, sg.d_y2, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
    CHECK_CUDA(cudaMemcpyAsync(sg.h_z2, sg.d_z2, (size_t)sg.segN*sizeof(float), cudaMemcpyDeviceToHost, gen.stream));
  }
  CHECK_CUDA(cudaEventRecord(sg.ready, gen.stream));
}

void orbit_gen_wait_slot(OrbitGenCtx& gen, int slot_idx){
  CHECK_CUDA(cudaSetDevice(gen.dev));
  CHECK_CUDA(cudaEventSynchronize(gen.slots[slot_idx].ready));
}

void orbit_gen_destroy(OrbitGenCtx& gen){
  CHECK_CUDA(cudaSetDevice(gen.dev));
  for(int slot=0; slot<2; ++slot){
    OrbitSegSlot &sg = gen.slots[slot];
    if(sg.ready) cudaEventDestroy(sg.ready);
    if(sg.h_z2) cudaFreeHost(sg.h_z2);
    if(sg.h_pos) cudaFreeHost(sg.h_pos);
    if(sg.h_y2) cudaFreeHost(sg.h_y2);
    if(sg.h_x2) cudaFreeHost(sg.h_x2);
    if(sg.h_z1) cudaFreeHost(sg.h_z1);
    if(sg.h_y1) cudaFreeHost(sg.h_y1);
    if(sg.h_x1) cudaFreeHost(sg.h_x1);
    if(sg.h_w) cudaFreeHost(sg.h_w);
    if(sg.h_v) cudaFreeHost(sg.h_v);
    if(sg.h_u) cudaFreeHost(sg.h_u);

    if(sg.d_z2) cudaFree(sg.d_z2);
    if(sg.d_y2) cudaFree(sg.d_y2);
    if(sg.d_x2) cudaFree(sg.d_x2);
    if(sg.d_z1) cudaFree(sg.d_z1);
    if(sg.d_y1) cudaFree(sg.d_y1);
    if(sg.d_x1) cudaFree(sg.d_x1);
    if(sg.d_w) cudaFree(sg.d_w);
    if(sg.d_v) cudaFree(sg.d_v);
    if(sg.d_u) cudaFree(sg.d_u);
    if(sg.d_pos) cudaFree(sg.d_pos);
  }

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
