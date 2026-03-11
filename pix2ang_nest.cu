#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
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

// --------- kernel code pasted here ----------
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

__global__ void pix2ang_nest_kernel(
    int nside,
    unsigned int base_ipix,
    int chunkN,
    float* __restrict__ theta,
    float* __restrict__ phi
) {
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
    float th = acosf(z);

    int jp = ( (c_jpll[face_num]*nr) + jpt + 1 + kshift ) >> 1;
    if (jp > nl4) jp -= nl4;
    if (jp < 1)   jp += nl4;

    float ph = (0.5f * (float)M_PI) * ( (float)jp - 0.5f*(float)(kshift + 1) ) / (float)nr;
    if (ph < 0.0f) ph += 2.0f * (float)M_PI;

    theta[tid] = th;
    phi[tid]   = ph;
}

// --------- simple CLI parse ----------
static inline std::string get_arg(int argc, char** argv, const std::string& key, const std::string& defv) {
  for (int i=1;i<argc;i++) {
    std::string s(argv[i]);
    if (s.rfind(key + "=", 0) == 0) return s.substr(key.size()+1);
  }
  return defv;
}
static inline long long to_ll(const std::string& s, long long defv){ try { return std::stoll(s); } catch(...) { return defv; } }
static inline int to_int(const std::string& s, int defv){ try { return std::stoi(s); } catch(...) { return defv; } }

int main(int argc, char** argv) {
    int dev = to_int(get_arg(argc, argv, "--gpu", "0"), 0);
    CHECK_CUDA(cudaSetDevice(dev));

    int nside = to_int(get_arg(argc, argv, "--nside", "0"), 0);
    long long npix_in = to_ll(get_arg(argc, argv, "--npix", "0"), 0);

    std::string tag = get_arg(argc, argv, "--tag", "10Mhz");
    std::string out_dir = get_arg(argc, argv, "--out_dir", ".");

    // 允许只给 npix：反推 nside
    if (nside <= 0) {
        if (npix_in <= 0) {
            std::cerr << "ERROR: provide --nside= or --npix=\n";
            return 1;
        }
        double ns = std::sqrt((double)npix_in / 12.0);
        nside = (int)llround(ns);
    }

    long long npix = 12LL * nside * (long long)nside;
    if (npix_in > 0 && npix_in != npix) {
        std::cerr << "WARN: npix_in("<<npix_in<<") != 12*nside^2("<<npix<<"), use computed npix="<<npix<<"\n";
    }

    std::cout << "gpu=" << dev << " nside=" << nside << " npix=" << npix << " tag="<<tag<<"\n";

    std::string ftheta = out_dir + "/theta_heal_" + tag + ".txt";
    std::string fphi   = out_dir + "/phi_heal_"   + tag + ".txt";

    std::ofstream ot(ftheta), op(fphi);
    if (!ot.is_open() || !op.is_open()) {
        std::cerr << "ERROR open output files:\n" << ftheta << "\n" << fphi << "\n";
        return 1;
    }
    // 大缓冲（重要：否则写巨文件会非常慢）
    static const size_t OUT_BUF_SZ = 16 << 20;
    static thread_local std::vector<char> buf1(OUT_BUF_SZ), buf2(OUT_BUF_SZ);
    ot.rdbuf()->pubsetbuf(buf1.data(), buf1.size());
    op.rdbuf()->pubsetbuf(buf2.data(), buf2.size());

    // chunk
    const int CHUNK_PIX = 1 << 20; // 1,048,576
    float *d_theta=nullptr, *d_phi=nullptr;
    CHECK_CUDA(cudaMalloc(&d_theta, (size_t)CHUNK_PIX*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_phi,   (size_t)CHUNK_PIX*sizeof(float)));

    float *h_theta=nullptr, *h_phi=nullptr;
    CHECK_CUDA(cudaMallocHost(&h_theta, (size_t)CHUNK_PIX*sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_phi,   (size_t)CHUNK_PIX*sizeof(float)));

    const int BLOCK = 256;

    for (long long base = 0; base < npix; base += CHUNK_PIX) {
        int cur = (int)std::min<long long>(CHUNK_PIX, npix - base);
        int grid = (cur + BLOCK - 1) / BLOCK;

        pix2ang_nest_kernel<<<grid, BLOCK>>>(
            nside, (unsigned int)base, cur, d_theta, d_phi
        );
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_theta, d_theta, (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_phi,   d_phi,   (size_t)cur*sizeof(float), cudaMemcpyDeviceToHost));

        for (int i=0;i<cur;i++) {
            ot << h_theta[i] << "\n";
            op << h_phi[i]   << "\n";
        }

        if ((base / CHUNK_PIX) % 32 == 0) {
            std::cout << "progress: " << (double)(base+cur) * 100.0 / (double)npix << "%\n";
        }
    }

    ot.close();
    op.close();

    CHECK_CUDA(cudaFreeHost(h_theta));
    CHECK_CUDA(cudaFreeHost(h_phi));
    CHECK_CUDA(cudaFree(d_theta));
    CHECK_CUDA(cudaFree(d_phi));

    std::cout << "Done. wrote:\n" << ftheta << "\n" << fphi << "\n";
    return 0;
}
