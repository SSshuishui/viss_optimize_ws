#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

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

struct Vec3 { float x, y, z; };

static inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline long long to_ll(const std::string& s, long long defv){ try { return std::stoll(s); } catch(...) { return defv; } }
static inline int to_int(const std::string& s, int defv){ try { return std::stoi(s); } catch(...) { return defv; } }
static inline float to_float(const std::string& s, float defv){ try { return std::stof(s); } catch(...) { return defv; } }

static std::vector<int> parse_tiles(const std::string& s) {
  std::vector<int> out;
  size_t st = 0;
  while (st < s.size()) {
    size_t ed = s.find(',', st);
    if (ed == std::string::npos) ed = s.size();
    std::string tok = s.substr(st, ed - st);
    if (!tok.empty()) out.push_back(std::max(1, std::stoi(tok)));
    st = ed + 1;
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

__device__ __forceinline__ float clampm1p1(float x) { return fminf(1.0f, fmaxf(-1.0f, x)); }
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

    z = clampm1p1(z);
    float th = acosf(z);

    int jp = ( (c_jpll[face_num]*nr) + jpt + 1 + kshift ) >> 1;
    if (jp > nl4) jp -= nl4;
    if (jp < 1)   jp += nl4;

    float ph = (0.5f * (float)M_PI) * ( (float)jp - 0.5f*(float)(kshift + 1) ) / (float)nr;
    if (ph < 0.0f) ph += 2.0f * (float)M_PI;

    float lat = (float)M_PI * 0.5f - th;
    if (ph > (float)M_PI) ph -= 2.0f * (float)M_PI;
    ph = -ph;

    float sl, cl, sp, cp;
    sincosf(lat, &sl, &cl);
    sincosf(ph, &sp, &cp);
    l[tid] = cl * cp;
    m[tid] = cl * sp;
    n[tid] = sl;
}

struct HistStats {
  double minv = 1e300;
  double maxv = -1e300;
  long double sum = 0.0L;
  long double sum2 = 0.0L;
  uint64_t count = 0;
  static constexpr double BIN_W = 1e-3; // degree
  static constexpr int NBINS = 180000;  // [0,180)
  std::vector<uint64_t> hist;
  HistStats() : hist(NBINS + 1, 0) {}
  void add(double deg) {
    minv = std::min(minv, deg);
    maxv = std::max(maxv, deg);
    sum += deg;
    sum2 += deg*deg;
    count++;
    int b = (int)std::floor(deg / BIN_W);
    if (b < 0) b = 0;
    if (b > NBINS) b = NBINS;
    hist[(size_t)b]++;
  }
  double mean() const { return count ? (double)(sum / (long double)count) : 0.0; }
  double stddev() const {
    if (count < 2) return 0.0;
    long double m = sum / (long double)count;
    long double v = sum2 / (long double)count - m*m;
    if (v < 0) v = 0;
    return sqrt((double)v);
  }
  double quantile(double q) const {
    if (count == 0) return 0.0;
    uint64_t target = (uint64_t)std::ceil(q * (double)count);
    if (target < 1) target = 1;
    uint64_t acc = 0;
    for (int i = 0; i <= NBINS; ++i) {
      acc += hist[(size_t)i];
      if (acc >= target) return (double)i * BIN_W;
    }
    return (double)NBINS * BIN_W;
  }
};

struct TileAnalyzer {
  int tile_pix;
  uint64_t max_tiles; // 0 => unlimited
  uint64_t tiles_done = 0;
  bool stop = false;

  std::vector<Vec3> buf;
  double sx = 0.0, sy = 0.0, sz = 0.0;
  HistStats alpha_deg_stats;
  HistStats lspan_stats, mspan_stats, nspan_stats;

  explicit TileAnalyzer(int t, uint64_t mt) : tile_pix(t), max_tiles(mt) {
    buf.reserve((size_t)tile_pix);
  }

  void finalize_tile() {
    if (buf.empty() || stop) return;
    double norm = sqrt(sx*sx + sy*sy + sz*sz);
    if (norm <= 0.0) {
      buf.clear(); sx = sy = sz = 0.0; return;
    }
    double cx = sx / norm, cy = sy / norm, cz = sz / norm;
    double max_ang = 0.0;
    float lmin = 1e30f, lmax = -1e30f;
    float mmin = 1e30f, mmax = -1e30f;
    float nmin = 1e30f, nmax = -1e30f;
    for (const auto& p : buf) {
      double dot = cx * (double)p.x + cy * (double)p.y + cz * (double)p.z;
      if (dot > 1.0) dot = 1.0;
      if (dot < -1.0) dot = -1.0;
      double ang = acos(dot);
      if (ang > max_ang) max_ang = ang;
      lmin = std::min(lmin, p.x); lmax = std::max(lmax, p.x);
      mmin = std::min(mmin, p.y); mmax = std::max(mmax, p.y);
      nmin = std::min(nmin, p.z); nmax = std::max(nmax, p.z);
    }
    alpha_deg_stats.add(max_ang * 180.0 / M_PI);
    lspan_stats.add((double)(lmax - lmin));
    mspan_stats.add((double)(mmax - mmin));
    nspan_stats.add((double)(nmax - nmin));
    tiles_done++;
    if (max_tiles > 0 && tiles_done >= max_tiles) stop = true;
    buf.clear(); sx = sy = sz = 0.0;
  }

  void add_point(float l, float m, float n) {
    if (stop) return;
    buf.push_back(Vec3{l,m,n});
    sx += (double)l; sy += (double)m; sz += (double)n;
    if ((int)buf.size() == tile_pix) finalize_tile();
  }

  void finish_all() {
    if (!stop && !buf.empty()) finalize_tile();
  }
};

static void print_summary(const TileAnalyzer& ta, double phi_deg) {
  const auto& A = ta.alpha_deg_stats;
  std::cout << "TILE_PIX=" << ta.tile_pix
            << " tiles=" << ta.tiles_done
            << "  alpha_deg[min/mean/p50/p90/p95/p99/max]=["
            << A.minv << "/" << A.mean() << "/"
            << A.quantile(0.50) << "/" << A.quantile(0.90) << "/"
            << A.quantile(0.95) << "/" << A.quantile(0.99) << "/"
            << A.maxv << "]"
            << "  alpha/phi(p99)=" << (phi_deg > 0 ? A.quantile(0.99)/phi_deg : 0.0)
            << "\n";

  std::cout << "           l_span[mean/p99/max]=[" << ta.lspan_stats.mean() << "/"
            << ta.lspan_stats.quantile(0.99) << "/" << ta.lspan_stats.maxv << "]"
            << "  m_span[mean/p99/max]=[" << ta.mspan_stats.mean() << "/"
            << ta.mspan_stats.quantile(0.99) << "/" << ta.mspan_stats.maxv << "]"
            << "  n_span[mean/p99/max]=[" << ta.nspan_stats.mean() << "/"
            << ta.nspan_stats.quantile(0.99) << "/" << ta.nspan_stats.maxv << "]"
            << "\n";
}

int main(int argc, char** argv) {
  int dev = to_int(get_arg(argc, argv, "--gpu", "0"), 0);
  CHECK_CUDA(cudaSetDevice(dev));

  int nside = to_int(get_arg(argc, argv, "--nside", "4096"), 4096);
  long long npix = 12LL * nside * (long long)nside;
  long long max_pix = to_ll(get_arg(argc, argv, "--max_pix", "0"), 0); // 0 => all
  if (max_pix > 0) npix = std::min(npix, max_pix);

  std::string tiles_s = get_arg(argc, argv, "--tiles", "64,128,256,512");
  std::vector<int> tile_sizes = parse_tiles(tiles_s);
  if (tile_sizes.empty()) {
    std::cerr << "ERROR --tiles empty\n";
    return 1;
  }
  uint64_t max_tiles = (uint64_t)to_ll(get_arg(argc, argv, "--max_tiles", "0"), 0); // per tile size
  int chunk_pix = to_int(get_arg(argc, argv, "--chunk_pix", std::to_string(1<<20)), 1<<20);

  float R = to_float(get_arg(argc, argv, "--R", "1737100"), 1737100.0f);
  float h = to_float(get_arg(argc, argv, "--h", "300000"), 300000.0f);
  float theta = asinf(R/(R+h));
  float phi = (float)M_PI - theta;
  double phi_deg = (double)phi * 180.0 / M_PI;

  std::cout << "gpu=" << dev << " nside=" << nside << " npix=" << npix
            << " phi_deg=" << phi_deg << " tiles=" << tiles_s
            << " max_tiles=" << max_tiles << " chunk_pix=" << chunk_pix << "\n";

  float *d_l=nullptr, *d_m=nullptr, *d_n=nullptr;
  float *h_l=nullptr, *h_m=nullptr, *h_n=nullptr;
  CHECK_CUDA(cudaMalloc(&d_l, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_n, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_l, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_m, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_n, (size_t)chunk_pix*sizeof(float)));

  std::vector<TileAnalyzer> analyzers;
  for (int t : tile_sizes) analyzers.emplace_back(t, max_tiles);

  const int BLOCK = 256;
  long long processed = 0;
  for (long long base = 0; base < npix; base += chunk_pix) {
    int cur = (int)std::min<long long>(chunk_pix, npix - base);
    int grid = (cur + BLOCK - 1) / BLOCK;
    pix2lmn_nest_kernel<<<grid, BLOCK>>>(nside, (unsigned int)base, cur, d_l, d_m, d_n);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_l, d_l, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_m, d_m, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_n, d_n, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));

    for (int i=0; i<cur; ++i) {
      float l = h_l[i], m = h_m[i], n = h_n[i];
      for (auto& ta : analyzers) ta.add_point(l,m,n);
    }
    processed += cur;

    if ((base / chunk_pix) % 8 == 0) {
      std::cout << "progress=" << (100.0 * (double)processed / (double)npix) << "%";
      for (const auto& ta : analyzers) {
        std::cout << "  [T=" << ta.tile_pix << ", tiles=" << ta.tiles_done << "]";
      }
      std::cout << "\n";
    }

    bool all_stop = true;
    for (const auto& ta : analyzers) if (!ta.stop) { all_stop = false; break; }
    if (all_stop) break;
  }

  for (auto& ta : analyzers) ta.finish_all();

  std::cout << "\n===== Tile locality summary =====\n";
  for (const auto& ta : analyzers) print_summary(ta, phi_deg);

  std::cout << "\nInterpretation:\n";
  std::cout << "- alpha_deg is the angular radius of a contiguous NEST tile on the unit sphere.\n";
  std::cout << "- Smaller alpha means tile-cone pruning is more likely to classify whole tiles as all-visible or all-hidden.\n";
  std::cout << "- Smaller TILE_PIX usually reduces alpha, but increases per-tile metadata and branch overhead.\n";
  std::cout << "- A good next step is to test TILE_PIX among 64/128/256 and then add orbit-aware tri-state sampling if needed.\n";

  CHECK_CUDA(cudaFreeHost(h_l));
  CHECK_CUDA(cudaFreeHost(h_m));
  CHECK_CUDA(cudaFreeHost(h_n));
  CHECK_CUDA(cudaFree(d_l));
  CHECK_CUDA(cudaFree(d_m));
  CHECK_CUDA(cudaFree(d_n));
  return 0;
}