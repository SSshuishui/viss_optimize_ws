// orbit_gen_seeded.cu
// Deterministic baseline/uvw generator matching your MATLAB MIncline + PosCreateUVW2 (day-level, unsegmented).
// Adds --seed for stable random window selection (t1 per orbit).
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_89 orbit_gen_nobll.cu -o orbit_gen_seeded   (4090)
//   nvcc -O3 -std=c++17 -arch=sm_86 orbit_gen_nobll.cu -o orbit_gen_seeded   (A6000)
//
// Run:
//   ./orbit_gen_seeded --only=10M --start=1 --end=1 --device=0 --seed=12345
//
// Notes:
// - This code generates ONE uvw/xyza/xyzb per day (same behavior as your current orbit_gen.cu).
// - Randomness (t1 selection) is deterministic given --seed and dayid.
// - It does NOT attempt to exactly match MATLAB's RNG stream; it guarantees reproducibility in CUDA/C++.

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// -------------------------
// CUDA error check
// -------------------------
#define CUDA_CHECK(call) do {                                     \
  cudaError_t err = (call);                                       \
  if (err != cudaSuccess) {                                       \
    fprintf(stderr, "CUDA error %s:%d: %s\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(err));         \
    std::exit(1);                                                 \
  }                                                               \
} while(0)

// -------------------------
// Config (match your MATLAB)
// -------------------------
static constexpr int   SATNUM = 8;
static constexpr float C0     = 3e8f;     // m/s
static constexpr float A_KM   = 2038.14f; // km
static constexpr float INCL   = 30.0f * (float)(M_PI / 180.0);
static constexpr float ARGP   = 0.0f;

// MATLAB: ProcessionCount = round(24/2.3)
static constexpr float ORBIT_HOURS = 2.3f;

// RAAN progression terms in your MATLAB
// raan = (g-1)*d2r*0.08 + (k/OrbitRes)*d2r*0.08 + (dayid-1)*d2r*0.8
static constexpr float RAAN_STEP_DEG_PER_REV = 0.08f;
static constexpr float RAAN_STEP_DEG_PER_K  = 0.08f;
static constexpr float RAAN_DAY_DEG         = 0.8f;

// -------------------------
// Baseline pair mapping in constant memory
// amount = satnum*(satnum-1)/2 = 28
// -------------------------
static constexpr int MAX_AMOUNT = 64;
__constant__ int d_pair_m[MAX_AMOUNT];
__constant__ int d_pair_n[MAX_AMOUNT];

// -------------------------
// Helpers (HOST)
// -------------------------

// MATLAB round: ties away from zero
static inline int round_away_from_zero(float x) {
  return (x >= 0.0f) ? (int)floorf(x + 0.5f) : (int)ceilf(x - 0.5f);
}

// MATLAB mod for int: result in [0, m-1]
static inline int mod_matlab_int(int a, int m) {
  int r = a % m;
  return (r < 0) ? (r + m) : r;
}

static void ensure_dir(const std::string& path) {
  std::filesystem::create_directories(path);
}

static void write_txt_3cols(const std::string& fn, const float* a, const float* b, const float* c, size_t n) {
  FILE* f = fopen(fn.c_str(), "wt");
  if (!f) { perror(("fopen " + fn).c_str()); std::exit(1); }
  static char buf[1<<20];
  setvbuf(f, buf, _IOFBF, sizeof(buf));
  for (size_t i=0;i<n;i++) {
    fprintf(f, "%.15f %.15f %.15f\n", (double)a[i], (double)b[i], (double)c[i]);
  }
  fclose(f);
}

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}

static int to_int(const std::string& s, int defv) {
  try { return std::stoi(s); } catch (...) { return defv; }
}

static uint64_t to_u64(const std::string& s, uint64_t defv) {
  try { return (uint64_t)std::stoull(s); } catch (...) { return defv; }
}

// Stable [0,1) double from rng (avoid uniform_real_distribution differences)
static inline double rand01(std::mt19937_64& rng) {
  const uint64_t r = rng();
  const uint64_t mant = r >> 11; // top 53 bits
  return (double)mant * (1.0 / 9007199254740992.0); // 2^53
}

// -------------------------
// Kernels (DEVICE)
// -------------------------

__device__ __forceinline__
float Mt_kf(int k0, int OrbitRes) {
  // MATLAB: Mt = linspace(1/OrbitRes, 2*pi+pi/200, OrbitRes);
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

  // e=0 fast path (as in your MATLAB: e=0)
  const float M = Mt_kf(k, OrbitRes);
  const float theta = M;
  const float r = A_KM;
  const float w = theta + ARGP;

  float sw, cw, sr, cr, si, ci;
  sincosf(w, &sw, &cw);
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
void k_norm3(const float* dx, const float* dy, const float* dz,
             float* missv, int n)
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
    const float rt = triangle_linspace_f(r2[sat], r1[sat], p, T); // meters

    const float ddx = rt * vx / mv / 1000.0f; // km
    const float ddy = rt * vy / mv / 1000.0f;
    const float ddz = rt * vz / mv / 1000.0f;

    x[base_ref + sat] = xr - ddx;
    y[base_ref + sat] = yr - ddy;
    z[base_ref + sat] = zr - ddz;
  }
}

__global__
void k_gather_pos(const float* x, const float* y, const float* z,
                  float* pos,
                  const int* t1s, int OrbitRes, int ProcessionCount, int segLen)
{
  const int outRow = blockIdx.x * blockDim.x + threadIdx.x;
  const int posnum = ProcessionCount * segLen;
  if (outRow >= posnum) return;

  const int orb = outRow / segLen;
  const int loc = outRow - orb * segLen;
  const int srcRow = t1s[orb] + loc;

  const int srcBase = srcRow * SATNUM;
  const int dstBase = outRow * (3 * SATNUM);

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

// -------------------------
// Main runner
// -------------------------
static void run_case(float frequency, const std::string& tag,
                     int start_day, int end_day, uint64_t seed)
{
  const float lambda_m  = C0 / frequency;

  const int OrbitRes = (int)std::ceil((double)(2.0 * M_PI) * (double)(100e3 / (double)lambda_m));
  const int ProcessionCount = round_away_from_zero(24.0f / ORBIT_HOURS);

  std::cout << "\n=== Run " << tag << "  freq=" << frequency
            << "  lambda=" << lambda_m
            << "  OrbitRes=" << OrbitRes
            << "  ProcessionCount=" << ProcessionCount
            << "  days=[" << start_day << "," << end_day << "]"
            << "  seed=" << seed
            << " ===\n";

  const std::string filepath = "./earth_" + tag + "hz/";
  ensure_dir(filepath);

  const float u0 = 100e3f / 23.0f;
  float h_r1[SATNUM] = {0, u0, 4*u0, 10*u0, 16*u0, 18*u0, 21*u0, 23*u0};
  float h_r2[SATNUM] = {0, 0.1e3f, 0.4f*u0, 1*u0, 1.6f*u0, 1.8f*u0, 2.1f*u0, 2.3f*u0};

  float *d_r1=nullptr, *d_r2=nullptr;
  CUDA_CHECK(cudaMalloc(&d_r1, SATNUM*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r2, SATNUM*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_r1, h_r1, SATNUM*sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_r2, h_r2, SATNUM*sizeof(float), cudaMemcpyHostToDevice));

  const int Nrows = OrbitRes * ProcessionCount;
  const int Nv = Nrows - 1;

  float *d_x=nullptr, *d_y=nullptr, *d_z=nullptr;
  CUDA_CHECK(cudaMalloc(&d_x, (size_t)Nrows * SATNUM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, (size_t)Nrows * SATNUM * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z, (size_t)Nrows * SATNUM * sizeof(float)));

  float *d_dx=nullptr, *d_dy=nullptr, *d_dz=nullptr, *d_missv=nullptr;
  CUDA_CHECK(cudaMalloc(&d_dx, (size_t)Nv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dy, (size_t)Nv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dz, (size_t)Nv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_missv, (size_t)Nv * sizeof(float)));

  const int segLen = OrbitRes / 3;
  const int posnum = ProcessionCount * segLen;

  int *d_t1s=nullptr;
  CUDA_CHECK(cudaMalloc(&d_t1s, (size_t)ProcessionCount * sizeof(int)));

  float *d_pos=nullptr;
  CUDA_CHECK(cudaMalloc(&d_pos, (size_t)posnum * (3*SATNUM) * sizeof(float)));

  const int amount = SATNUM*(SATNUM-1)/2;
  const int outN = 2 * amount * posnum;

  float *d_u=nullptr,*d_v=nullptr,*d_w=nullptr;
  float *d_x1=nullptr,*d_y1=nullptr,*d_z1=nullptr;
  float *d_x2=nullptr,*d_y2=nullptr,*d_z2=nullptr;

  CUDA_CHECK(cudaMalloc(&d_u,  (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v,  (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_w,  (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x1, (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y1, (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z1, (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_x2, (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y2, (size_t)outN * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_z2, (size_t)outN * sizeof(float)));

  std::vector<float> h_u(outN), h_v(outN), h_w(outN);
  std::vector<float> h_x1(outN), h_y1(outN), h_z1(outN);
  std::vector<float> h_x2(outN), h_y2(outN), h_z2(outN);

  const int BS = 256;

  for (int dayid = start_day; dayid <= end_day; ++dayid) {
    std::cout << "dayid=" << dayid << " (" << tag << ")\n";

    uint64_t day_seed = seed ^ (uint64_t)dayid * 0x9E3779B97F4A7C15ULL;
    std::mt19937_64 rng(day_seed);

    CUDA_CHECK(cudaMemset(d_x, 0, (size_t)Nrows * SATNUM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)Nrows * SATNUM * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_z, 0, (size_t)Nrows * SATNUM * sizeof(float)));

    // 1) sat1 xyz
    {
      const int GS = (Nrows + BS - 1) / BS;
      k_compute_sat1_xyz<<<GS, BS>>>(d_x, d_y, d_z, OrbitRes, ProcessionCount, dayid);
      CUDA_CHECK(cudaGetLastError());
    }

    // 2) diff + norm
    {
      const int GS = (Nv + BS - 1) / BS;
      k_diff_sat1<<<GS, BS>>>(d_x, d_y, d_z, d_dx, d_dy, d_dz, OrbitRes, ProcessionCount);
      CUDA_CHECK(cudaGetLastError());
      k_norm3<<<GS, BS>>>(d_dx, d_dy, d_dz, d_missv, Nv);
      CUDA_CHECK(cudaGetLastError());
    }

    // 3) other satellites
    {
      const int T = round_away_from_zero((float)OrbitRes * 7.0f * 24.0f / ORBIT_HOURS);
      const int mod14 = mod_matlab_int(dayid - 1, 14);
      const int day_offset = round_away_from_zero((float)mod14 * (float)OrbitRes * 24.0f / ORBIT_HOURS);

      const int GS = (Nv + BS - 1) / BS;
      k_compute_other_sats<<<GS, BS>>>(d_x, d_y, d_z,
                                       d_dx, d_dy, d_dz, d_missv,
                                       d_r1, d_r2,
                                       OrbitRes, ProcessionCount,
                                       T, day_offset);
      CUDA_CHECK(cudaGetLastError());
    }

    // 4) choose per-orbit random window t1 and gather pos
    {
      std::vector<int> h_t1s(ProcessionCount);
      for (int g=0; g<ProcessionCount; ++g) {
        const int orbitStart = g * OrbitRes;
        const int orbitEnd   = (g+1)*OrbitRes - 1;

        double u01 = rand01(rng);
        int t1 = orbitStart + round_away_from_zero((float)(u01 * (double)OrbitRes * (2.0/3.0)));
        int t2 = t1 + segLen - 1;

        if (t2 > orbitEnd) {
          t2 = orbitEnd;
          t1 = t2 - segLen + 1;
        }
        if (t1 < orbitStart) t1 = orbitStart;
        if (t1 + segLen - 1 > orbitEnd) t1 = orbitEnd - segLen + 1;

        h_t1s[g] = t1;
      }
      CUDA_CHECK(cudaMemcpy(d_t1s, h_t1s.data(), (size_t)ProcessionCount*sizeof(int), cudaMemcpyHostToDevice));

      const int GS = (posnum + BS - 1) / BS;
      k_gather_pos<<<GS, BS>>>(d_x, d_y, d_z, d_pos, d_t1s, OrbitRes, ProcessionCount, segLen);
      CUDA_CHECK(cudaGetLastError());
    }

    // 5) uvw/xyz
    {
      const int GS = (outN + BS - 1) / BS;
      k_compute_uvw_xyz<<<GS, BS>>>(d_pos, posnum, lambda_m,
                                    d_u, d_v, d_w,
                                    d_x1, d_y1, d_z1,
                                    d_x2, d_y2, d_z2);
      CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(h_u.data(),  d_u,  (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_v.data(),  d_v,  (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_w.data(),  d_w,  (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x1.data(), d_x1, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y1.data(), d_y1, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z1.data(), d_z1, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_x2.data(), d_x2, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y2.data(), d_y2, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_z2.data(), d_z2, (size_t)outN*sizeof(float), cudaMemcpyDeviceToHost));

    char fn1[512], fn2[512], fn3[512];
    snprintf(fn1, sizeof(fn1), "%suvw%dday%s.txt",  filepath.c_str(), dayid, tag.c_str());
    snprintf(fn2, sizeof(fn2), "%sxyza%dday%s.txt", filepath.c_str(), dayid, tag.c_str());
    snprintf(fn3, sizeof(fn3), "%sxyzb%dday%s.txt", filepath.c_str(), dayid, tag.c_str());

    write_txt_3cols(fn1, h_u.data(),  h_v.data(),  h_w.data(),  (size_t)outN);
    write_txt_3cols(fn2, h_x1.data(), h_y1.data(), h_z1.data(), (size_t)outN);
    write_txt_3cols(fn3, h_x2.data(), h_y2.data(), h_z2.data(), (size_t)outN);
  }

  CUDA_CHECK(cudaFree(d_r1));
  CUDA_CHECK(cudaFree(d_r2));
  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_z));
  CUDA_CHECK(cudaFree(d_dx));
  CUDA_CHECK(cudaFree(d_dy));
  CUDA_CHECK(cudaFree(d_dz));
  CUDA_CHECK(cudaFree(d_missv));
  CUDA_CHECK(cudaFree(d_t1s));
  CUDA_CHECK(cudaFree(d_pos));
  CUDA_CHECK(cudaFree(d_u));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_w));
  CUDA_CHECK(cudaFree(d_x1));
  CUDA_CHECK(cudaFree(d_y1));
  CUDA_CHECK(cudaFree(d_z1));
  CUDA_CHECK(cudaFree(d_x2));
  CUDA_CHECK(cudaFree(d_y2));
  CUDA_CHECK(cudaFree(d_z2));
}

// -------------------------
// Main
// -------------------------
int main(int argc, char** argv) {
  // copy baseline mapping once
  {
    std::vector<int> pair_m, pair_n;
    pair_m.reserve(SATNUM*(SATNUM-1)/2);
    pair_n.reserve(SATNUM*(SATNUM-1)/2);
    for (int m=0;m<SATNUM-1;m++) {
      for (int n=m+1;n<SATNUM;n++) {
        pair_m.push_back(m);
        pair_n.push_back(n);
      }
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_pair_m, pair_m.data(), pair_m.size()*sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_pair_n, pair_n.data(), pair_n.size()*sizeof(int)));
  }

  int device = to_int(get_arg(argc, argv, "--device", "0"), 0);
  CUDA_CHECK(cudaSetDevice(device));

  int start_day = to_int(get_arg(argc, argv, "--start", "1"), 1);
  int end_day   = to_int(get_arg(argc, argv, "--end",   "1"), 1);
  if (end_day < start_day) std::swap(start_day, end_day);

  std::string only = get_arg(argc, argv, "--only", "all");
  uint64_t seed = to_u64(get_arg(argc, argv, "--seed", "12345"), 1234567ULL);

  if (only == "1M" || only == "all")  run_case(1e6f, "1M",  start_day, end_day, seed);
  if (only == "10M"|| only == "all")  run_case(1e7f, "10M", start_day, end_day, seed);
  if (only == "30M"|| only == "all")  run_case(3e7f, "30M", start_day, end_day, seed);

  return 0;
}