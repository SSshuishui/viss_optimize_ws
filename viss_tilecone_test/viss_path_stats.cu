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
#include <random>
#include <cstdint>
#include <omp.h>
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

static inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline int to_int(const std::string& s, int defv){ try { return std::stoi(s); } catch(...) { return defv; } }
static inline long long to_ll(const std::string& s, long long defv){ try { return std::stoll(s); } catch(...) { return defv; } }
static inline uint64_t to_u64(const std::string& s, uint64_t defv){ try { return (uint64_t)std::stoull(s); } catch(...) { return defv; } }

struct Vec3f { float x,y,z; };
struct TileMeta { float cx, cy, cz, alpha; };

static inline int round_away_from_zero_host(float x) {
  return (x >= 0.0f) ? (int)floorf(x + 0.5f) : (int)ceilf(x - 0.5f);
}
static inline int mod_matlab_int_host(int a, int m) {
  int r = a % m;
  return (r < 0) ? (r + m) : r;
}
__host__ __device__ static inline float clampm1p1(float x){ return fminf(1.0f, fmaxf(-1.0f, x)); }

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

static void add_tile_range(const std::vector<float>& l, const std::vector<float>& m, const std::vector<float>& n,
                           int begin, int end, std::vector<TileMeta>& out, int tile_stride, uint64_t max_tiles)
{
  int tile_id = begin / (end-begin+1); // unused placeholder
  for(int p = begin; p < end; p += (end-begin+1)) { (void)p; }
}

static std::vector<TileMeta> build_sampled_tiles_gpu(int dev, int nside, long long pix0, long long pix1, int tile_pix,
                                                     int chunk_pix, int tile_stride, uint64_t max_tiles)
{
  CHECK_CUDA(cudaSetDevice(dev));
  float *d_l=nullptr,*d_m=nullptr,*d_n=nullptr;
  float *h_l=nullptr,*h_m=nullptr,*h_n=nullptr;
  CHECK_CUDA(cudaMalloc(&d_l, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_m, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMalloc(&d_n, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_l, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_m, (size_t)chunk_pix*sizeof(float)));
  CHECK_CUDA(cudaMallocHost(&h_n, (size_t)chunk_pix*sizeof(float)));

  const int BLOCK = 256;
  std::vector<TileMeta> metas;
  std::vector<Vec3f> tilebuf;
  tilebuf.reserve((size_t)tile_pix);
  long long tile_idx_global = 0;
  int need = tile_pix;
  bool keep = ((tile_idx_global % tile_stride) == 0);

  auto finalize_tile = [&](void){
    if(keep && !tilebuf.empty()){
      double sx=0,sy=0,sz=0;
      for(auto &v: tilebuf){ sx+=v.x; sy+=v.y; sz+=v.z; }
      double inv = 1.0 / sqrt(sx*sx + sy*sy + sz*sz + 1e-30);
      double cx = sx*inv, cy = sy*inv, cz = sz*inv;
      double maxa = 0.0;
      for(auto &v: tilebuf){
        double dot = cx*v.x + cy*v.y + cz*v.z;
        if(dot > 1.0) dot = 1.0;
        if(dot < -1.0) dot = -1.0;
        double a = acos(dot);
        if(a > maxa) maxa = a;
      }
      metas.push_back(TileMeta{(float)cx,(float)cy,(float)cz,(float)maxa});
    }
    tilebuf.clear();
    tile_idx_global++;
    keep = ((tile_idx_global % tile_stride) == 0);
    need = tile_pix;
  };

  for(long long base = pix0; base < pix1; base += chunk_pix){
    int cur = (int)std::min<long long>(chunk_pix, pix1 - base);
    int grid = (cur + BLOCK - 1) / BLOCK;
    pix2lmn_nest_kernel<<<grid, BLOCK>>>(nside, (unsigned int)base, cur, d_l, d_m, d_n);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_l, d_l, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_m, d_m, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_n, d_n, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));

    for(int i=0;i<cur;i++){
      if(keep) tilebuf.push_back(Vec3f{h_l[i],h_m[i],h_n[i]});
      need--;
      if(need == 0){
        finalize_tile();
        if(max_tiles > 0 && metas.size() >= max_tiles) goto done;
      }
    }
  }
  if(need != tile_pix){
    finalize_tile();
  }
done:
  CHECK_CUDA(cudaFreeHost(h_l)); CHECK_CUDA(cudaFreeHost(h_m)); CHECK_CUDA(cudaFreeHost(h_n));
  CHECK_CUDA(cudaFree(d_l)); CHECK_CUDA(cudaFree(d_m)); CHECK_CUDA(cudaFree(d_n));
  return metas;
}

static inline float Mt_kf(int k0, int OrbitRes) {
  const float start = 1.0f / (float)OrbitRes;
  const float end   = (float)(2.0 * M_PI + M_PI / 200.0);
  if (OrbitRes <= 1) return start;
  const float t = (float)k0 / (float)(OrbitRes - 1);
  return start + (end - start) * t;
}

static void generate_all_positions_cpu(int dayid, float lambda_m, std::vector<Vec3f>& posrows, int& OrbitRes, int& ProcessionCount, int& segLen){
  static constexpr int SATNUM = 8;
  static constexpr float A_KM   = 2038.14f;
  static constexpr float INCL   = 30.0f * (float)(M_PI / 180.0);
  static constexpr float ARGP   = 0.0f;
  static constexpr float ORBIT_HOURS = 2.3f;
  static constexpr float RAAN_STEP_DEG_PER_REV = 0.08f;
  static constexpr float RAAN_STEP_DEG_PER_K   = 0.08f;
  static constexpr float RAAN_DAY_DEG          = 0.8f;
  float r1[SATNUM] = {0, 100e3f/23.0f, 4*(100e3f/23.0f), 10*(100e3f/23.0f), 16*(100e3f/23.0f), 18*(100e3f/23.0f), 21*(100e3f/23.0f), 23*(100e3f/23.0f)};
  float r2[SATNUM] = {0, 0.1e3f, 0.4f*(100e3f/23.0f), 1*(100e3f/23.0f), 1.6f*(100e3f/23.0f), 1.8f*(100e3f/23.0f), 2.1f*(100e3f/23.0f), 2.3f*(100e3f/23.0f)};

  OrbitRes = (int)std::ceil((double)(2.0 * M_PI) * (double)(100e3 / (double)lambda_m));
  ProcessionCount = round_away_from_zero_host(24.0f / ORBIT_HOURS);
  segLen = OrbitRes / 3;
  const int Nrows = OrbitRes * ProcessionCount;
  posrows.assign((size_t)Nrows * SATNUM, Vec3f{0,0,0});

  for(int idx=0; idx<Nrows; ++idx){
    int g = idx / OrbitRes;
    int k = idx - g * OrbitRes;
    float d2r = (float)(M_PI / 180.0);
    float raan = (float)g * d2r * RAAN_STEP_DEG_PER_REV + ((float)(k+1)/(float)OrbitRes) * d2r * RAAN_STEP_DEG_PER_K + (float)(dayid-1) * d2r * RAAN_DAY_DEG;
    float M = Mt_kf(k, OrbitRes);
    float theta = M;
    float r = A_KM;
    float ww = theta + ARGP;
    float sw,cw,sr,cr,si,ci;
    sincosf(ww,&sw,&cw); sincosf(raan,&sr,&cr); sincosf(INCL,&si,&ci);
    float X = r * (cw*cr - sw*ci*sr);
    float Y = r * (cw*sr + sw*ci*cr);
    float Z = r * (sw*si);
    posrows[(size_t)idx*SATNUM + 0] = {X,Y,Z};
  }
  std::vector<float> dx(Nrows-1), dy(Nrows-1), dz(Nrows-1), missv(Nrows-1);
  for(int i=0;i<Nrows-1;++i){
    auto a = posrows[(size_t)i*SATNUM + 0];
    auto b = posrows[(size_t)(i+1)*SATNUM + 0];
    dx[i]=b.x-a.x; dy[i]=b.y-a.y; dz[i]=b.z-a.z;
    missv[i] = sqrtf(dx[i]*dx[i]+dy[i]*dy[i]+dz[i]*dz[i]);
  }
  int T = round_away_from_zero_host((float)OrbitRes * 7.0f * 24.0f / ORBIT_HOURS);
  int mod14 = mod_matlab_int_host(dayid - 1, 14);
  int day_offset = round_away_from_zero_host((float)mod14 * (float)OrbitRes * 24.0f / ORBIT_HOURS);
  for(int i=0;i<Nrows-1;++i){
    int period = 2*T;
    int p = (period>0) ? ((day_offset+i)%period) : 0;
    float vx=dx[i], vy=dy[i], vz=dz[i], mv=missv[i]+1e-20f;
    int row = i+1;
    Vec3f ref = posrows[(size_t)row*SATNUM + 0];
    for(int sat=1;sat<SATNUM;++sat){
      float rt;
      if(p < T){ float tt = (T<=1)?0.f:(float)p/(float)(T-1); rt = r2[sat] + (r1[sat]-r2[sat])*tt; }
      else { int q = p-T; float tt = (T<=1)?0.f:(float)q/(float)(T-1); rt = r1[sat] + (r2[sat]-r1[sat])*tt; }
      float ddx = rt * vx / mv / 1000.0f;
      float ddy = rt * vy / mv / 1000.0f;
      float ddz = rt * vz / mv / 1000.0f;
      posrows[(size_t)row*SATNUM + sat] = {ref.x - ddx, ref.y - ddy, ref.z - ddz};
    }
  }
}

static std::vector<int> make_t1s(int dayid, int OrbitRes, int ProcessionCount, int segLen, uint64_t seed){
  std::vector<int> h_t1s(ProcessionCount);
  uint64_t day_seed = seed ^ (uint64_t)dayid * 0x9E3779B97F4A7C15ULL;
  std::mt19937_64 rng(day_seed);
  for (int g=0; g<ProcessionCount; ++g) {
    int orbitStart = g * OrbitRes;
    int orbitEnd   = (g+1)*OrbitRes - 1;
    double u01 = (double)(rng() >> 11) * (1.0 / 9007199254740992.0);
    int t1 = orbitStart + round_away_from_zero_host((float)(u01 * (double)OrbitRes * (2.0/3.0)));
    int t2 = t1 + segLen - 1;
    if (t2 > orbitEnd) { t2 = orbitEnd; t1 = t2 - segLen + 1; }
    if (t1 < orbitStart) t1 = orbitStart;
    if (t1 + segLen - 1 > orbitEnd) t1 = orbitEnd - segLen + 1;
    h_t1s[g] = t1;
  }
  return h_t1s;
}

int main(int argc, char** argv){
  int dev = to_int(get_arg(argc, argv, "--gpu", "0"), 0);
  CHECK_CUDA(cudaSetDevice(dev));
  std::string btag = get_arg(argc, argv, "--btag", "10M");
  int nside = to_int(get_arg(argc, argv, "--nside", (btag=="30M"?"16384":(btag=="10M"?"4096":"512"))), 4096);
  int day_start = to_int(get_arg(argc, argv, "--day_start", "1"), 1);
  int day_count = to_int(get_arg(argc, argv, "--day_count", "1"), 1);
  int segs = to_int(get_arg(argc, argv, "--segs", (btag=="30M"?"10":(btag=="10M"?"10":"1"))), 10);
  int tile_pix = to_int(get_arg(argc, argv, "--viss_tile_pix", "256"), 256);
  int tile_stride = to_int(get_arg(argc, argv, "--stats_tile_stride", "1"), 1);
  uint64_t max_tiles = to_u64(get_arg(argc, argv, "--stats_max_tiles", "0"), 0);
  uint64_t orbit_seed = to_u64(get_arg(argc, argv, "--orbit_seed", "42"), 42ULL);
  int chunk_pix = to_int(get_arg(argc, argv, "--chunk_pix", std::to_string(1<<20)), 1<<20);

  long long npix = 12LL * (long long)nside * (long long)nside;
  float frequency=(btag=="1M")? 1e6f : (btag=="10M"? 1e7f : 3e7f);
  float lamda=3e8f/frequency;
  float R=1737.1e3f, h=300e3f;
  float theta=asinf(R/(R+h));
  float phi=(float)M_PI-theta;
  float cosphi=cosf(phi);
  std::cout << "gpu=" << dev << " btag=" << btag << " nside=" << nside << " npix=" << npix
            << " tile_pix=" << tile_pix << " tile_stride=" << tile_stride << " max_tiles=" << max_tiles
            << " phi_deg=" << (phi*180.0/M_PI) << "\n";

  // build sampled tile meta once
  auto metas = build_sampled_tiles_gpu(dev, nside, 0, npix, tile_pix, chunk_pix, tile_stride, max_tiles);
  std::cout << "sampled_tiles=" << metas.size() << "\n";

  for(int dayid = day_start; dayid < day_start + day_count; ++dayid){
    int OrbitRes=0, ProcessionCount=0, segLen=0;
    std::vector<Vec3f> posrows;
    generate_all_positions_cpu(dayid, lamda, posrows, OrbitRes, ProcessionCount, segLen);
    auto t1s = make_t1s(dayid, OrbitRes, ProcessionCount, segLen, orbit_seed);
    int T = ProcessionCount * segLen;
    if(segs > T) segs = T;
    std::vector<int> seg_t0(segs), seg_tlen(segs);
    int baseT = T / segs, remT = T % segs, curT=0;
    for(int k=0;k<segs;k++){ int len = baseT + (k<remT?1:0); seg_t0[k]=curT; seg_tlen[k]=len; curT+=len; }
    std::cout << "===== Day " << dayid << " / " << (day_start+day_count-1) << " =====\n";

    static const int pair_m[28] = {0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,4,4,4,5,5,6};
    static const int pair_n[28] = {1,2,3,4,5,6,7,2,3,4,5,6,7,3,4,5,6,7,4,5,6,7,5,6,7,6,7,7};

    for(int s=0;s<segs;s++){
      int t0 = seg_t0[s], tlen = seg_tlen[s];
      std::vector<Vec3f> satdir((size_t)tlen * 8);
      for(int tl=0; tl<tlen; ++tl){
        int outRowGlobal = t0 + tl;
        int orb = outRowGlobal / segLen;
        int loc = outRowGlobal - orb * segLen;
        int srcRow = t1s[orb] + loc;
        for(int sat=0;sat<8;++sat){
          Vec3f p = posrows[(size_t)srcRow*8 + sat];
          float inv = 1.0f / sqrtf(p.x*p.x + p.y*p.y + p.z*p.z);
          satdir[(size_t)tl*8 + sat] = {p.x*inv, p.y*inv, p.z*inv};
        }
      }

      uint64_t hide=0,vv=0,vm=0,mv=0,mm=0;
      double t0s = omp_get_wtime();
      #pragma omp parallel
      {
        uint64_t lhide=0, lvv=0, lvm=0, lmv=0, lmm=0;
        #pragma omp for schedule(static)
        for(long long tt=0; tt<(long long)tlen * (long long)metas.size(); ++tt){
          int tl = (int)(tt / (long long)metas.size());
          int ti = (int)(tt - (long long)tl * (long long)metas.size());
          TileMeta tm = metas[(size_t)ti];
          unsigned char st[8];
          for(int sat=0;sat<8;++sat){
            Vec3f sd = satdir[(size_t)tl*8 + sat];
            float dot = tm.cx*sd.x + tm.cy*sd.y + tm.cz*sd.z;
            dot = clampm1p1(dot);
            float delta = acosf(dot);
            if(delta + tm.alpha <= phi) st[sat] = 2;           // V
            else if(delta - tm.alpha > phi) st[sat] = 0;       // H
            else st[sat] = 1;                                  // M
          }
          for(int k=0;k<28;++k){
            unsigned char a = st[pair_m[k]], b = st[pair_n[k]];
            if(a==0 || b==0) lhide++;
            else if(a==2 && b==2) lvv++;
            else if(a==2 && b==1) lvm++;
            else if(a==1 && b==2) lmv++;
            else lmm++;
          }
        }
        #pragma omp atomic
        hide += lhide;
        #pragma omp atomic
        vv += lvv;
        #pragma omp atomic
        vm += lvm;
        #pragma omp atomic
        mv += lmv;
        #pragma omp atomic
        mm += lmm;
      }
      double dt = omp_get_wtime() - t0s;
      long double total = (long double)hide + vv + vm + mv + mm;
      auto pct = [&](uint64_t v)->double{ return total>0 ? (double)((long double)v*100.0L/total) : 0.0; };
      std::cout << "[day "<<dayid<<" seg "<<(s+1)<<"/"<<segs<<"] t0="<<t0
                << " tlen="<<tlen
                << " segN_half="<<(tlen*28)
                << " StatsTime="<<dt<<"s"
                << " sampled_tiles="<<metas.size()
                << " path[hide/vv/vm/mv/mm]=["<<hide<<"/"<<vv<<"/"<<vm<<"/"<<mv<<"/"<<mm<<"]"
                << " ratio[%]=["<<pct(hide)<<"/"<<pct(vv)<<"/"<<pct(vm)<<"/"<<pct(mv)<<"/"<<pct(mm)<<"]\n";
    }
  }
  return 0;
}
