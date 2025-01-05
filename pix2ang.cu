#include <cstdio>
#include <iostream>
#include <ctime>
#include <string>
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "error.cuh"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>

#define _USE_MATH_DEFINES

using namespace std;
using Complex = thrust::complex<float>;

// complexExp 函数的实现
__device__ thrust::complex<float> complexExp(const Complex &d) {
    float realPart = exp(d.real()) * cos(d.imag());
    float imagPart = exp(d.real()) * sin(d.imag());
    return thrust::complex<float>(realPart, imagPart);
}
// complexAbs 函数的实现
__device__ thrust::complex<float> ComplexAbs(const Complex &d) {
    // 复数的模定义为 sqrt(real^2 + imag^2)
    return thrust::complex<float>(sqrt(d.real() * d.real() + d.imag() * d.imag()));
}

struct timeval start, finish;
float total_time;

void writeToFile(const thrust::device_vector<float>& device_vector, const std::string& filename) {
    // 将数据从设备内存复制到主机内存
    std::vector<float> host_vector(device_vector.size());
    thrust::copy(device_vector.begin(), device_vector.end(), host_vector.begin());
    // 打开文件
    std::ofstream file(filename);
    if (file.is_open()) {
        // 按照指定格式写入文件
        for(const float& value : host_vector)
        {
            // file << value.real() << " " << value.imag() << std::endl;
            file << value << std::endl;
        }
    }
    // 关闭文件
    file.close();
}

// mk_pix2xy 函数
__global__ void mk_pix2xy(int *pix2x, int *pix2y) 
{
    int kpix = blockIdx.x * blockDim.x + threadIdx.x;
    if (kpix >= 1024) return;

    int jpix = kpix;
    int ix = 0;
    int iy = 0;
    int ip = 1;

    while (true) {
        if (jpix == 0)
            break;
        // bit value (in kpix), goes in ix
        int id = jpix % 2;
        jpix /= 2;
        ix = id * ip + ix;
        // bit value (in kpix), goes in iy
        id = jpix % 2;
        jpix /= 2;
        iy = id * ip + iy;

        // next bit (in x and y)
        ip *= 2;
    }
    pix2x[kpix] = ix;
    pix2y[kpix] = iy;
}


// f_pix2and_nest 函数
__global__ void f_pix2and_nest(
            int *face_num, int *ipf, int *ip_low, 
            int *ix, int *iy, int *pix2x, int *pix2y, 
            int *jrt, int *jpt, float *z, float *kshift, float *nr, 
            int *jr, int *jp, float *theta, float *phi, 
            int *jrll, int *jpll, int nside, int nl4, int npface, float fact1, float fact2) 
{
    int ipix = blockIdx.x * blockDim.x + threadIdx.x;

    // face number in {0,11}
    face_num[ipix] = ipix / npface;
    // pixel number in the face {0,npface-1}
    ipf[ipix] = ipix % npface;

    int scalemlv = 1;
    for(int i=0; i<=4; i++){
        ip_low[ipix] = ipf[ipix] % 1024;
        ix[ipix] = truncf(ix[ipix] + scalemlv * pix2x[ip_low[ipix]+1]);
        iy[ipix] = truncf(iy[ipix] + scalemlv * pix2y[ip_low[ipix]+1]);
        scalemlv = scalemlv * 32;
        ipf[ipix] = ipf[ipix]/1024;
    }
    
    ix[ipix] = truncf(ix[ipix] + scalemlv * pix2x[ipf[ipix]+1]);
    iy[ipix] = truncf(iy[ipix] + scalemlv * pix2y[ipf[ipix]+1]);

    // transforms to (horizontal, vertical) coordinates
    jrt[ipix] = truncf(ix[ipix] + iy[ipix]);
    jpt[ipix] = truncf(ix[ipix] - iy[ipix]);

    // jr =  fix(jrll(face_num+1)*nSide - jrt - 1);
    jr[ipix] = truncf(jrll[face_num[ipix] + 1] * nside - jrt[ipix] - 1);

    // compute z coordinate on the sphere
    // north pole region
    if (jr[ipix] < nside){
        float nrM = truncf(jr[ipix]);
        z[ipix] = 1 - nrM * nrM * fact1;
        kshift[ipix] = 0;
        nr[ipix] = nrM;
    }
    // equatorial region
    else if (jr[ipix] >= nside && jr[ipix] <= 3 * nside){
        z[ipix] = (2 * nside - jr[ipix]) * fact2;
        kshift[ipix] = (jr[ipix]-nside) % 2;
        nr[ipix] = static_cast<float>(nside);
    }
    // south pole region
    else if(jr[ipix] > 3*nside){
        float nrM = truncf(nl4 - jr[ipix]);
        z[ipix] = -1 + nrM * nrM * fact1;
        kshift[ipix] = 0;
        nr[ipix] = nrM;
    }

    // Convert z to theta
    theta[ipix] = acosf(z[ipix]);

    // Compute phi in [0,2*M_PI)
    jp[ipix] = truncf((jpll[face_num[ipix] + 1] * nr[ipix] + jpt[ipix] + 1 + kshift[ipix]) / 2);
    if (jp[ipix] > nl4){
        jp[ipix] = truncf(jp[ipix] - nl4);
    }
    if (jp[ipix] < 1){
        jp[ipix] = truncf(jp[ipix] + nl4);
    }

    phi[ipix] = (M_PI / 2.0f) * (jp[ipix] - (kshift[ipix] + 1) / 2.0f) / nr[ipix];
}


int vissGen() 
{   
    // 读取 B.txt 文件
    string address_B = "B.txt";
    ifstream BFile;
    BFile.open(address_B);
    int npix = 0;
    BFile >> npix;  // 读取第一行的数据，也就是总数据行数
    cout << "npix: " << npix << endl;
    vector<float> B(npix);
    for (int i = 0; i < npix; i++) {
        BFile >> B[i];
    }
    BFile.close();
    
    int nside=round(sqrt(npix/12));
    float s=4*M_PI/npix;
    float res=sqrt(4*M_PI/npix);

    // f_pix2and_nest 函数调用，获得每个点的
    thrust::device_vector<int> jrll = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    thrust::device_vector<int> jpll = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};
    
    // 对应 [pix2x,pix2y] = mk_pix2xy;
    thrust::device_vector<int> pix2x(1024);
    thrust::device_vector<int> pix2y(1024);

    int blockSize;
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, mk_pix2xy, 0, 0);
    int gridSize = floor(1024 + blockSize - 1) / blockSize;
    cout << "pix2xy Computing, blockSize: " << blockSize << endl;
    cout << "pix2xy Computing, girdSize: " << gridSize << endl;
    mk_pix2xy<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(pix2x.data()), 
        thrust::raw_pointer_cast(pix2y.data())
    );
    // 进行线程同步
    CHECK(cudaDeviceSynchronize());

    int npface = nside * nside;
    int nl4 = 4 * nside;
    float fact1 = 1.0f / (3.0f * nside * nside);
    float fact2 = 2.0f / (3.0f * nside);

    cout<< "res: " << res << endl;
    cout<< "s: " << s << endl;
    cout<< "nside: " << nside << endl;
    cout<< "npface: " << npface << endl;
    cout<< "nl4: " << nl4 << endl;
    cout<< "fact1: " << fact1 << endl;
    cout<< "fact2: " << fact2 << endl;


    // find face and pixel number in the face
    thrust::device_vector<int> face_num(npix);
    thrust::device_vector<int> ipf(npix);

    thrust::device_vector<int> ip_low(npix);
    thrust::device_vector<int> ix(npix);
    thrust::device_vector<int> iy(npix);
    thrust::device_vector<int> jrt(npix);
    thrust::device_vector<int> jrp(npix);
    thrust::device_vector<float> z(npix);
    thrust::device_vector<float> kshift(npix);
    thrust::device_vector<float> nr(npix);

    thrust::device_vector<int> jr(npix);
    thrust::device_vector<int> jp(npix);

    // 核函数返回的结果 坐标点 theta_heal, phi_heal
    thrust::device_vector<float> theta_heal(npix);
    thrust::device_vector<float> phi_heal(npix);

    // 调用核函数
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, f_pix2and_nest, 0, 0);
    gridSize = (npix + blockSize - 1) / blockSize;
    cout << "f_pix2and_nest computing, blockSize: " << blockSize << endl;
    cout << "f_pix2and_nest computing, girdSize: " << gridSize << endl;
    f_pix2and_nest<<<gridSize, blockSize>>>(
        thrust::raw_pointer_cast(face_num.data()),
        thrust::raw_pointer_cast(ipf.data()),
        thrust::raw_pointer_cast(ip_low.data()),
        thrust::raw_pointer_cast(ix.data()),
        thrust::raw_pointer_cast(iy.data()),
        thrust::raw_pointer_cast(pix2x.data()),
        thrust::raw_pointer_cast(pix2y.data()),
        thrust::raw_pointer_cast(jrt.data()),
        thrust::raw_pointer_cast(jrp.data()),
        thrust::raw_pointer_cast(z.data()),
        thrust::raw_pointer_cast(kshift.data()),
        thrust::raw_pointer_cast(nr.data()),
        thrust::raw_pointer_cast(jr.data()),
        thrust::raw_pointer_cast(jp.data()),
        thrust::raw_pointer_cast(theta_heal.data()),
        thrust::raw_pointer_cast(phi_heal.data()),
        thrust::raw_pointer_cast(jrll.data()),
        thrust::raw_pointer_cast(jpll.data()),
        nside, nl4, npface, fact1, fact2
    );
    // 进行线程同步
    CHECK(cudaDeviceSynchronize());

    writeToFile(theta_heal, "theta_heal.txt");
    writeToFile(phi_heal, "phi_heal.txt");

    return 0;
}


int main()
{
    vissGen();
}

