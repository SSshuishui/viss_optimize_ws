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
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/unique.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>


#define _USE_MATH_DEFINES
#define EXP 0.0000000000

using namespace std;
using Complex = thrust::complex<float>;
const int uvw_presize = 400000;


// complexExp 函数的实现
__device__ thrust::complex<float> complexExp(Complex d) {
    return thrust::exp(d);
}

// complexAbs 函数的实现
__device__ thrust::complex<float> ComplexAbs(const Complex &d) {
    // 复数的模定义为 sqrt(real^2 + imag^2)
    return thrust::complex<float>(sqrt(d.real() * d.real() + d.imag() * d.imag()));
}

__device__ float norm(float x, float y, float z) {
    return sqrtf(x * x + y * y + z * z);
}


struct timeval start, finish;
float total_time;

string address = "./earth_1Mhz/";

__global__ void healpix_moonback_pre(
    float * __restrict__ theta_heal, 
    float * __restrict__ phi_heal,
    float * __restrict__ l, 
    float * __restrict__ m, 
    float * __restrict__ n,
    float * __restrict__ B, 
    int npix, 
    float s)
{
    // 获取线程索引
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= npix) return;

    // 预先加载频繁使用的数据到寄存器
    float theta_val = theta_heal[index];
    float phi_val = phi_heal[index];

    // 优化角度计算
    theta_val = M_PI / 2 - theta_val;
    if (phi_val > M_PI) {
        phi_val -= 2 * M_PI;
    }
    phi_val = -phi_val;

    // 计算l, m, n
    float cos_theta = cosf(theta_val);
    l[index] = cos_theta * cosf(phi_val);
    m[index] = cos_theta * sinf(phi_val);
    n[index] = sinf(theta_val);

    // 更新theta_heal和phi_heal
    theta_heal[index] = theta_val;
    phi_heal[index] = phi_val;

    // 缩放B
    B[index] *= s;
}

// 启动核函数的包装函数
void launch_healpix_moonback_pre(
    float *d_theta_heal, 
    float *d_phi_heal,
    float *d_l, 
    float *d_m, 
    float *d_n,
    float *d_B, 
    int npix, 
    float s)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, healpix_moonback_pre, 0, 0);
    int blocksPerGrid = floor(npix + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    healpix_moonback_pre<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_theta_heal, d_phi_heal, d_l, d_m, d_n, d_B, npix, s);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void healpix_moonback_viss(
    float * __restrict__ B, 
    Complex * __restrict__ Viss,
    float * __restrict__ u, 
    float * __restrict__ v, 
    float * __restrict__ w,
    float * __restrict__ xyz1a, 
    float * __restrict__ xyz1b, 
    float * __restrict__ xyz1c, 
    float * __restrict__ xyz2a, 
    float * __restrict__ xyz2b, 
    float * __restrict__ xyz2c,
    float * __restrict__ l, 
    float * __restrict__ m, 
    float * __restrict__ n, 
    int amount, 
    int npix, 
    float phi,
    Complex zero, 
    Complex I1, 
    Complex two, 
    Complex CPI)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i >= amount) return;

    // 预先加载频繁使用的数据到寄存器
    float u_val = u[i];
    float v_val = v[i];
    float w_val = w[i];
    float xyz1a_val = xyz1a[i];
    float xyz1b_val = xyz1b[i];
    float xyz1c_val = xyz1c[i];
    float xyz2a_val = xyz2a[i];
    float xyz2b_val = xyz2b[i];
    float xyz2c_val = xyz2c[i];
    float norm1 = norm(xyz1a_val, xyz1b_val, xyz1c_val);
    float norm2 = norm(xyz2a_val, xyz2b_val, xyz2c_val);

    Complex acc = zero;
    for (int index = 0; index < npix; index++) {
        // 天空每个点与视场中心的夹角
        float gb1_comp = l[index] * xyz1a_val + m[index] * xyz1b_val + n[index] * xyz1c_val;
        float gb2_comp = l[index] * xyz2a_val + m[index] * xyz2b_val + n[index] * xyz2c_val;
        float beta1 = acosf(gb1_comp / norm1);
        float beta2 = acosf(gb2_comp / norm2);

        if (beta1 <= phi && beta2 <= phi) {
            float phase = u_val * l[index] + v_val * m[index] + w_val * (n[index] - 1.0f);
            Complex vari(phase, 0.0f);
            acc += Complex(B[index], 0) * thrust::exp((zero - I1) * two * CPI * vari);
        }
    }

    Viss[i] = acc;
    Viss[i + amount] = thrust::conj(acc);
}

// 启动核函数的包装函数
void launch_healpix_moonback_viss(
    float *d_B, 
    Complex *d_Viss,
    float *d_u, 
    float *d_v, 
    float *d_w,
    float *d_xyz1a, 
    float *d_xyz1b, 
    float *d_xyz1c, 
    float *d_xyz2a, 
    float *d_xyz2b, 
    float *d_xyz2c,
    float *d_l, 
    float *d_m, 
    float *d_n, 
    int amount, 
    int npix, 
    float phi,
    Complex zero, 
    Complex I1, 
    Complex two, 
    Complex CPI)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, healpix_moonback_viss, 0, 0);
    int blocksPerGrid = floor(amount + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    healpix_moonback_viss<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_B, d_Viss, d_u, d_v, d_w, d_xyz1a, d_xyz1b, d_xyz1c, 
        d_xyz2a, d_xyz2b, d_xyz2c, d_l, d_m, d_n, amount, npix, phi, 
        zero, I1, two, CPI);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}

            
__host__ float calculate_fa(const thrust::device_vector<float>& u,
                            const thrust::device_vector<float>& v,
                            const thrust::device_vector<float>& w) 
{
    float sum_v2 = thrust::transform_reduce(v.begin(), v.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());
    float sum_u_w = thrust::inner_product(u.begin(), u.end(), w.begin(), 0.0f);
    float sum_u_v = thrust::inner_product(u.begin(), u.end(), v.begin(), 0.0f);
    float sum_v_w = thrust::inner_product(v.begin(), v.end(), w.begin(), 0.0f);
    float sum_u2 = thrust::transform_reduce(u.begin(), u.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());

    return (sum_v2 * sum_u_w - sum_u_v * sum_v_w) / (sum_u2 * sum_v2 - sum_u_v * sum_u_v);
}


__host__ float calculate_fb(const thrust::device_vector<float>& u,
                            const thrust::device_vector<float>& v,
                            const thrust::device_vector<float>& w) 
{
    float sum_v2 = thrust::transform_reduce(v.begin(), v.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());
    float sum_u_w = thrust::inner_product(u.begin(), u.end(), w.begin(), 0.0f);
    float sum_u_v = thrust::inner_product(u.begin(), u.end(), v.begin(), 0.0f);
    float sum_v_w = thrust::inner_product(v.begin(), v.end(), w.begin(), 0.0f);
    float sum_u2 = thrust::transform_reduce(u.begin(), u.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());

    return (sum_u2 * sum_v_w - sum_u_v * sum_u_w) / (sum_u2 * sum_v2 - sum_u_v * sum_u_v);
}


__global__ void viss_trans(
    Complex* __restrict__ Viss,
    float* __restrict__ w,
    int size,
    Complex zero,
    Complex I1,
    Complex two,
    Complex CPI)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // gViss = gViss * exp(-1i * 2 * M_PI * gw)
    Complex cw(w[idx], 0.0);
    Viss[idx] = Viss[idx] * complexExp((zero-I1) * two * CPI * cw);
}

// 启动核函数的包装函数
void launch_viss_trans(
    Complex *d_Viss,
    float *d_w,
    int size,
    Complex zero,
    Complex I1,
    Complex two,
    Complex CPI)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, viss_trans, 0, 0);
    int blocksPerGrid = floor(size + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    viss_trans<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_Viss, d_w, size, zero, I1, two, CPI
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void meshgrid_kernel(
    float* __restrict__ ug,
    float* __restrict__ vg,
    int RES,
    float start,
    float step)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= RES * RES) return;

    int i = idx / RES;
    int j = idx % RES;
    ug[idx] = start + i * step;
    vg[idx] = start + j * step;
}

// 启动核函数的包装函数
void launch_meshgrid_kernel(
    float *d_ug,
    float *d_vg,
    int RES,
    float start,
    float step)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, meshgrid_kernel, 0, 0);
    int blocksPerGrid = floor(RES * RES + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 启动核函数
    meshgrid_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_ug, d_vg, RES, start, step
    );
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void round_divide_kernel(
    float* __restrict__ vec,
    float du,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    vec[idx] = roundf(vec[idx] / du);
}

// 启动核函数的包装函数
void launch_round_divide_kernel(
    float *d_vec,
    float du,
    int size)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, round_divide_kernel, 0, 0);
    int blocksPerGrid = floor(size + threadsPerBlock - 1) / threadsPerBlock;
    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // 启动核函数
    round_divide_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_vec, du, size
    );
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void calculate_locg(
    float* __restrict__ u,
    float* __restrict__ v,
    int* __restrict__ locg,
    int n,
    int RES)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    locg[idx] = (u[idx] + (RES - 1) / 2) * RES + (v[idx] + (RES - 1) / 2) + 1;
}

// 启动核函数的包装函数
void launch_calculate_locg(
    float *d_u,
    float *d_v,
    int *d_locg,
    int n,
    int RES)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, calculate_locg, 0, 0);
    int blocksPerGrid = floor(n + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    calculate_locg<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_u, d_v, d_locg, n, RES
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


// 计算特定 q 值的索引有几个
__global__ void computeLocCount(
    int* __restrict__ il,
    int* __restrict__ countLoc,
    int nulocg,
    int uvw_index)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nulocg) return;

    // countLoc: (nulocg, 1)
    int count = 0;
    for (int i = 0; i < uvw_index; i++) {
        if (il[i] == q) {
            count++;
        }
    }
    countLoc[q] = count;
}

// 启动核函数的包装函数
void launch_computeLocCount(
    int *d_il,
    int *d_countLoc,
    int nulocg,
    int uvw_index)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, computeLocCount, 0, 0);
    int blocksPerGrid = floor(nulocg + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    computeLocCount<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_il, d_countLoc, nulocg, uvw_index
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


// 把每个 q 值对应的索引保存下来
__global__ void computeLocViss(
    int* __restrict__ il,
    int* __restrict__ ilq,
    int* __restrict__ countLoc,
    int nulocg,
    int uvw_index)
{
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= nulocg) return;

    // ilq: (uvw_index, 1)
    // countLoc: (nulocg, 1)
    int start_idx=0;
    for(int i=1; i<=q; i++){
        start_idx += countLoc[i-1];
    }
    for (int i = 0; i < uvw_index; i++) {
        if (il[i] == q) {
            ilq[start_idx] = i;
            start_idx++;
        }
    }
}

// 启动核函数的包装函数
void launch_computeLocViss(
    int *d_il,
    int *d_ilq,
    int *d_countLoc,
    int nulocg,
    int uvw_index)
{
    // 计算网格和块的大小
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, computeLocViss, 0, 0);
    int blocksPerGrid = floor(nulocg + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    computeLocViss<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_il, d_ilq, d_countLoc, nulocg, uvw_index
    );

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 销毁流
    cudaStreamDestroy(stream);
}


__global__ void computeC(
    int npix,
    float* __restrict__ ug,
    float* __restrict__ vg,
    Complex* __restrict__ Viss,
    int* __restrict__ il,
    int* __restrict__ ulocg,
    int* __restrict__ locg,
    Complex* __restrict__ C,
    float* __restrict__ l,
    float* __restrict__ m,
    float* __restrict__ n,
    int* __restrict__ ilq,
    int* __restrict__ countLoc,
    int nulocg,
    float fa,
    float fb,
    int uvw_index,
    Complex two,
    Complex I1,
    Complex CPI)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= npix) return;

    // 预先加载频繁使用的数据到寄存器
    float l_val = l[idx];
    float m_val = m[idx];
    float n_val = n[idx];

    Complex acc = C[idx];

    int start_idx = 0;
    for (int q=0; q<nulocg; q++) {
        // ugu=ug(ulocg(q));  vgu=vg(ulocg(q));
        int loc = ulocg[q];
        float ugu = ug[loc-1];
        float vgu = vg[loc-1];
        
        // Find indices where il == q
        // 传入一个ilq数组，用来存储所有等于q的索引，维度是il的维度(uvw_index)
        // 每个索引的数量countLoc  维度是nulocg
        Complex Vissgu(0.0, 0.0);
        for(int con=0; con < countLoc[q]; con++){
            int locViss = ilq[con + start_idx];
            Vissgu += Viss[locViss];
        }
        start_idx += countLoc[q];
        Vissgu = Vissgu / Complex(countLoc[q], 0.0f);   // 一个数值

        Complex Vissp = Vissgu * complexExp(two * CPI * I1 * Complex((ugu * fa + vgu * fb) * n_val, 0.0));
        acc += Vissp * complexExp(two * I1 * CPI * Complex(ugu * l_val + vgu * m_val, 0.0));
    }
    C[idx] = acc;
}

// 启动核函数的包装函数
void launch_computeC(
    int npix,
    float *d_ug,
    float *d_vg,
    Complex *d_Viss,
    int *d_il,
    int *d_ulocg,
    int *d_locg,
    Complex *d_C,
    float *d_l,
    float *d_m,
    float *d_n,
    int *d_ilq,
    int *d_countLoc,
    int nulocg,
    float fa,
    float fb,
    int uvw_index,
    Complex two,
    Complex I1,
    Complex CPI)
{
    int threadsPerBlock;
    int minGridSize; // 最小网格大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &threadsPerBlock, computeC, 0, 0);
    int blocksPerGrid = floor(npix + threadsPerBlock - 1) / threadsPerBlock;

    // 创建CUDA流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 启动核函数
    computeC<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        npix, d_ug, d_vg, d_Viss, d_il, d_ulocg, d_locg, d_C, d_l, d_m, d_n, d_ilq, d_countLoc, nulocg, fa, fb, uvw_index, two, I1, CPI
    );
    
    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err!= cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    // 销毁流
    cudaStreamDestroy(stream);
}


// 优化后的版本
int vissGen(float frequency) 
{   
    gettimeofday(&start, NULL);

    int nDevices=1;
    // 设置节点数量（gpu显卡数量）
    // CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    cout << "frequency: " << frequency << endl;

    int days = 2;
    Complex I1(0.0, 1.0);
    Complex zero(0.0, 0.0);
    Complex one(1.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);
    cout << "days: " << days << endl;

    // 读取 B.txt, theta_heal.txt, phi_heal.txt 文件
    string address_B = address + "B.txt";
    string address_theta_heal = address + "theta_heal.txt";
    string address_phi_heal = address + "phi_heal.txt";
    ifstream BFile, thetaFile, phiFile;
    BFile.open(address_B);
    thetaFile.open(address_theta_heal);
    phiFile.open(address_phi_heal);

    int npix = 0;
    BFile >> npix;  // 读取第一行的数据，也就是总数据行数
    cout << "npix: " << npix << endl;
    
    vector<float> cB(npix), ctheta_heal(npix), cphi_heal(npix);
    for (int i = 0; i < npix; i++) {
        BFile >> cB[i];
        thetaFile >> ctheta_heal[i];
        phiFile >> cphi_heal[i];
    }
    BFile.close();
    thetaFile.close();
    phiFile.close();

    cout << "load B.txt, theta_heal.txt, phi_heal.txt in CPU success" << endl;
    
    int nside=round(sqrt(npix/12));
    float s=4*M_PI/npix;
    float res=sqrt(4*M_PI/npix);

    // f_pix2and_nest 函数调用，获得每个点的
    thrust::device_vector<int> jrll = {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4};
    thrust::device_vector<int> jpll = {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7};

    float R=1737.1e3;  // 月球半径
    float h=300e3;     // 卫星轨道高度
    float theta= asinf(R/(R+h));  // 单颗卫星的月背遮挡区的半视场角
    float phi= M_PI-theta;   // 每条基线的半视场角，每条基线对应一个半视场角
    cout << "theta: " << theta << endl;
    cout << "phi: " << phi << endl;

    // recon的参数
    float bl_max=100e3;
    float lamda = 3e8 / frequency;
    float nr=ceil(bl_max/lamda*2); // 环的数量 半个波长一个环

    cout << "nside: " << nside << endl;
    cout << "s: " << s << endl;
    cout << "res: " << res << endl;
    cout << "theta: " << theta << endl;
    cout << "lamda: " << lamda << endl;
    cout << "nr: " << nr << endl;


    // 开启cpu线程并行
    // 一个线程处理1个GPU
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(tid);
        std::cout << "Thread " << tid << " is running on device " << tid << endl;

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid; p < days; p += nDevices) {
            cout << "for loop: " << p+1 << endl;

            // 将 B, theta_heal, phi_heal 数据从CPU搬到GPU上        
            thrust::device_vector<float> B(cB.begin(), cB.end());
            thrust::device_vector<float> theta_heal(ctheta_heal.begin(), ctheta_heal.end());
            thrust::device_vector<float> phi_heal(cphi_heal.begin(), cphi_heal.end());

            thrust::device_vector<float> l(npix), m(npix), n(npix);

            std::vector<float> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
            thrust::device_vector<float> u(uvw_presize), v(uvw_presize), w(uvw_presize);

            std::vector<float> cxyz1a(uvw_presize), cxyz1b(uvw_presize), cxyz1c(uvw_presize);
            thrust::device_vector<float> xyz1a(uvw_presize), xyz1b(uvw_presize), xyz1c(uvw_presize);

            std::vector<float> cxyz2a(uvw_presize), cxyz2b(uvw_presize), cxyz2c(uvw_presize);
            thrust::device_vector<float> xyz2a(uvw_presize), xyz2b(uvw_presize), xyz2c(uvw_presize);

            std::vector<float> cbll(uvw_presize);
            thrust::device_vector<float> bll(uvw_presize);

            // std::vector<Complex> cViss(uvw_presize);
            thrust::device_vector<Complex> Viss(uvw_presize);

            // 存储最终的计算结果
            thrust::device_vector<Complex> C(npix);

            cudaEvent_t compute_start, compute_stop;
            cudaEventCreate(&compute_start);
            cudaEventCreate(&compute_stop);
            cudaEventRecord(compute_start);

            int uvw_index, xyz1_index, xyz2_index, bll_index; 
            #pragma omp critical
            {   
                // 读取 uvw
                string address_uvw = address + "updated_uvw" + to_string(p+1) + "day1M.txt";
                cout << "address_uvw: " << address_uvw << endl;
                ifstream uvwFile(address_uvw);
                uvw_index = 0;
                float u_point, v_point, w_point, f_point;
                if (uvwFile.is_open()) {
                    uvwFile >> u_point >> v_point >> w_point >> f_point; // 读取第一行，删除
                    while (uvwFile >> u_point >> v_point >> w_point >> f_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cu[uvw_index] = u_point;
                        cv[uvw_index] = v_point;
                        cw[uvw_index] = w_point;
                        uvw_index++;
                    }
                }
                cout << "uvw_index: " << uvw_index << endl;
                // 复制到GPU上
                thrust::copy(cu.begin(), cu.begin() + uvw_index, u.begin());
                thrust::copy(cv.begin(), cv.begin() + uvw_index, v.begin());
                thrust::copy(cw.begin(), cw.begin() + uvw_index, w.begin());
                
                // 读取 xyz1(xyza)
                string address_xyz1 = address + "xyza" + to_string(p+1) + "day1M.txt";
                cout << "address_xyz1: " << address_xyz1 << endl;
                ifstream xyz1File(address_xyz1);
                xyz1_index = 0;
                float a_point, b_point, c_point;
                if (xyz1File.is_open()) {
                    xyz1File >> a_point >> b_point >> c_point;
                    while (xyz1File >> a_point >> b_point >> c_point) {
                        cxyz1a[xyz1_index] = a_point;
                        cxyz1b[xyz1_index] = b_point;
                        cxyz1c[xyz1_index] = c_point;
                        xyz1_index++;
                    }
                }
                cout << "xyz1_index: " << xyz1_index << endl;
                // 复制到GPU上
                thrust::copy(cxyz1a.begin(), cxyz1a.begin() + xyz1_index, xyz1a.begin());
                thrust::copy(cxyz1b.begin(), cxyz1b.begin() + xyz1_index, xyz1b.begin());
                thrust::copy(cxyz1c.begin(), cxyz1c.begin() + xyz1_index, xyz1c.begin());

                // 读取 xyz2(xyzb)
                string address_xyz2 = address + "xyzb" + to_string(p+1) + "day1M.txt";
                cout << "address_xyz2: " << address_xyz2 << endl;
                ifstream xyz2File(address_xyz2);
                xyz2_index = 0;
                if (xyz2File.is_open()) {
                    xyz2File >> a_point >> b_point >> c_point;
                    while (xyz2File >> a_point >> b_point >> c_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cxyz2a[xyz2_index] = a_point;
                        cxyz2b[xyz2_index] = b_point;
                        cxyz2c[xyz2_index] = c_point;
                        xyz2_index++;
                    }
                }
                cout << "xyz2_index: " << xyz2_index << endl;
                // 复制到GPU上
                thrust::copy(cxyz2a.begin(), cxyz2a.begin() + xyz2_index, xyz2a.begin());
                thrust::copy(cxyz2b.begin(), cxyz2b.begin() + xyz2_index, xyz2b.begin());
                thrust::copy(cxyz2c.begin(), cxyz2c.begin() + xyz2_index, xyz2c.begin());

                // 读取 bll
                string address_bll = address + "bll" + to_string(p+1) + "day1M.txt";
                cout << "address_bll: " << address_bll << endl;
                ifstream bllFile(address_bll);
                bll_index = 0;
                if (bllFile.is_open()) {
                    bllFile >> a_point;
                    while (bllFile >> a_point) {
                        // cu, cv, cw 需要存储原始坐标
                        cbll[bll_index] = a_point;
                        bll_index++;
                    }
                }
                cout << "bll_index: " << bll_index << endl;
                // 复制到GPU上
                thrust::copy(cbll.begin(), cbll.begin() + bll_index, bll.begin());
            }

            // 计算可见度
            int amount = ceil(uvw_index/2);
            cout << "amount: " << amount << endl;

            launch_healpix_moonback_pre(
                thrust::raw_pointer_cast(theta_heal.data()), 
                thrust::raw_pointer_cast(phi_heal.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(B.data()),
                npix, s);
            CHECK(cudaDeviceSynchronize());  


            printf("Viss Computing... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_healpix_moonback_viss(
                thrust::raw_pointer_cast(B.data()),
                thrust::raw_pointer_cast(Viss.data()),
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(w.data()),
                thrust::raw_pointer_cast(xyz1a.data()),
                thrust::raw_pointer_cast(xyz1b.data()),
                thrust::raw_pointer_cast(xyz1c.data()),
                thrust::raw_pointer_cast(xyz2a.data()),
                thrust::raw_pointer_cast(xyz2b.data()),
                thrust::raw_pointer_cast(xyz2c.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                amount, npix, phi,
                zero, I1, two, CPI);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " Viss Computing Success!" << endl;
            for (int i=0; i<=2; i++){
                cout << "Viss[" << i << "]: " << Viss[i] << endl;
            }
             

            // 图像重构
            // fa=(sum(v.^2)*sum(u.*w)-sum(u.*v)*sum(v.*w))/(sum(u.^2)*sum(v.^2)-sum(u.*v)^2);
            // fb=(sum(u.^2)*sum(v.*w)-sum(u.*v)*sum(u.*w))/(sum(u.^2)*sum(v.^2)-sum(u.*v)^2);
            float fa = calculate_fa(u, v, w);
            float fb = calculate_fb(u, v, w);

            // 计算 lmax, mmax 和 lmmax
            float lmax = std::sqrt(1 + fa * fa);
            float mmax = std::sqrt(1 + fb * fb);
            float lmmax = std::max(lmax, mmax);

            // Viss 去除相位
            // gViss = gViss.*exp(-1i*2*pi*gw);  
            launch_viss_trans(
                thrust::raw_pointer_cast(Viss.data()),
                thrust::raw_pointer_cast(w.data()),
                uvw_index, zero, two, CPI, I1);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " viss trans..." << endl;

            // 计算 RES
            float temp_res = std::ceil(2 * bl_max / lamda * 2 * lmmax);
            int RES = static_cast<int>(temp_res + 2 + 1 - std::fmod(temp_res, 2));

            // 计算 cug
            thrust::host_vector<float> h_cug(RES);
            float start = -bl_max / lamda;
            float end = bl_max / lamda;
            float step = (end - start) / (RES - 1);
            for (int i = 0; i < RES; ++i) {
                h_cug[i] = start + i * step;
            }
            thrust::device_vector<float> cug = h_cug;

            // 计算 du
            float du = 2 * bl_max / lamda / (RES - 1);

            // 创建 ug 和 vg 的网格
            thrust::device_vector<float> ug(RES * RES);
            thrust::device_vector<float> vg(RES * RES);

            // 调用 CUDA 内核创建网格
            printf("meshgrid... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_meshgrid_kernel(
                thrust::raw_pointer_cast(ug.data()), 
                thrust::raw_pointer_cast(vg.data()), 
                RES, start, step);
            CHECK(cudaDeviceSynchronize());

            // 四舍五入 u 和 v
            launch_round_divide_kernel(thrust::raw_pointer_cast(u.data()), du, u.size());
            launch_round_divide_kernel(thrust::raw_pointer_cast(v.data()), du, v.size());
            CHECK(cudaDeviceSynchronize());

            // 计算 
            thrust::device_vector<int> locg(uvw_index);
            printf("locg... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_calculate_locg(
                thrust::raw_pointer_cast(u.data()),
                thrust::raw_pointer_cast(v.data()),
                thrust::raw_pointer_cast(locg.data()),
                uvw_index, RES
            );
            CHECK(cudaDeviceSynchronize());

            thrust::device_vector<int> ulocg = locg;
            // 排序
            thrust::sort(ulocg.begin(), ulocg.end());
            // 找到唯一元素并复制到 ulocg
            auto end_ind = thrust::unique(ulocg.begin(), ulocg.end());
            ulocg.resize(end_ind - ulocg.begin());

            // 初始化 il 设备向量
            thrust::device_vector<int> il(locg.size());

            // 填充 il 设备向量
            thrust::lower_bound(ulocg.begin(), ulocg.end(), locg.begin(), locg.end(), il.begin());

            // 计算 C
            // 记录imagerecon开始事件
            cudaEvent_t imagereconstart, imagereconstop;
            cudaEventCreate(&imagereconstart);
            cudaEventCreate(&imagereconstop);
            cudaEventRecord(imagereconstart);

            int nulocg = ulocg.size();
            cout << "nulocg: " << nulocg << endl;

            // 先提前计算每个 q 值的索引有几个
            thrust::device_vector<int> countLoc(nulocg);
            launch_computeLocCount(
                thrust::raw_pointer_cast(il.data()), 
                thrust::raw_pointer_cast(countLoc.data()), 
                nulocg, uvw_index);
            CHECK(cudaDeviceSynchronize());

            // 然后存下来每个 q 值对应的索引
            thrust::device_vector<int> ilq(il.size());
            printf("Compute LocViss... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_computeLocViss(
                thrust::raw_pointer_cast(il.data()), 
                thrust::raw_pointer_cast(ilq.data()), 
                thrust::raw_pointer_cast(countLoc.data()), 
                nulocg, uvw_index);
            CHECK(cudaDeviceSynchronize());

            // 计算的时候直接根据 q 对应的索引个数加载索引值
            printf("Compute C... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            launch_computeC(
                npix,
                thrust::raw_pointer_cast(ug.data()), 
                thrust::raw_pointer_cast(vg.data()), 
                thrust::raw_pointer_cast(Viss.data()),
                thrust::raw_pointer_cast(il.data()),
                thrust::raw_pointer_cast(ulocg.data()),
                thrust::raw_pointer_cast(locg.data()),
                thrust::raw_pointer_cast(C.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(ilq.data()),
                thrust::raw_pointer_cast(countLoc.data()),
                nulocg, fa, fb,
                uvw_index, two, I1, CPI);
            CHECK(cudaDeviceSynchronize());
            cout << "Period " << p+1 << " compute C success" << endl;

            // 记录imagerecon结束事件
            cudaEventRecord(imagereconstop);
            cudaEventSynchronize(imagereconstop);
            // 计算经过的时间
            float imagereconMS = 0;
            cudaEventElapsedTime(&imagereconMS, imagereconstart, imagereconstop);
            printf("Period %d Image Reconstruction Cost Time is: %f s\n", p+1, imagereconMS/1000);
            // 销毁事件
            cudaEventDestroy(imagereconstart);
            cudaEventDestroy(imagereconstop);

            // 记录compute结束事件
            cudaEventRecord(compute_stop);
            cudaEventSynchronize(compute_stop);
            // 计算经过的时间
            float computeMS = 0;
            cudaEventElapsedTime(&computeMS, compute_start, compute_stop);
            printf("Period %d Compute Cost Time is: %f s\n", p+1, computeMS/1000);
            // 销毁事件
            cudaEventDestroy(compute_start);
            cudaEventDestroy(compute_stop);

            for (int i=0; i<=2; i++){
                cout << "C[" << i << "]: " << C[i] << endl;
            }
            

            // 创建一个临界区，用于保存结果
            #pragma omp critical
            {   
                // 将数据从设备内存复制到主机内存
                std::vector<Complex> host_C(C.size());
                CHECK(cudaMemcpy(host_C.data(), thrust::raw_pointer_cast(C.data()), C.size() * sizeof(Complex), cudaMemcpyDeviceToHost));
                CHECK(cudaDeviceSynchronize());
                // 打开文件
                string address_C = "wsnoblockage/C" + to_string(p+1) + "day1M.txt";
                cout << "Period " << p+1 << " save address_C: " << address_C << endl;
                std::ofstream file(address_C);
                if (file.is_open()) {
                    // 按照指定格式写入文件
                    for(const Complex& value : host_C)
                    {
                        file << value.real() << std::endl;
                    }
                }
                // 关闭文件
                file.close();
                std::cout << "Period " << p+1 << " save C success!" << std::endl;
            }
        }
    }
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{
    vissGen(1e7);
}

