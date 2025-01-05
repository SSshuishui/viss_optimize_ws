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
const int uvw_presize = 4000000;


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


__global__ void healpix_moonback_pre(float *theta_heal, float *phi_heal,
                                float *l, float *m, float *n,
                                float *B, int npix, float s)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < npix) {
        // Convert theta_heal
        theta_heal[index] = M_PI / 2 - theta_heal[index];
        
        // Modify phi_heal
        if (phi_heal[index] > M_PI) {
            phi_heal[index] -= 2 * M_PI;
        }
        phi_heal[index] = -phi_heal[index];
        // Compute n, l, m
        l[index] = cosf(theta_heal[index]) * cosf(phi_heal[index]);
        m[index] = cosf(theta_heal[index]) * sinf(phi_heal[index]);
        n[index] = sinf(theta_heal[index]);

        B[index] = B[index] * s;
    }
}

__global__ void healpix_moonback_viss(float *B, Complex *Viss,
                                float *u, float *v, float *w,
                                float *xyz1a, float *xyz1b, float *xyz1c, 
                                float *xyz2a, float *xyz2b, float *xyz2c,
                                float *l, float *m, float *n, 
                                int amount, int npix, float phi,
                                Complex zero, Complex I1, Complex two, Complex CPI)  
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if (i < amount) {
        Viss[i] = zero;
        for (int index = 0; index < npix; index++){
            // 天空每个点与视场中心的夹角
            float gb1_comp = l[index]*xyz1a[i] + m[index]*xyz1b[i] + n[index]*xyz1c[i];
            float gb2_comp = l[index]*xyz2a[i] + m[index]*xyz2b[i] + n[index]*xyz2c[i];
            float beta1 = acosf(gb1_comp / norm(xyz1a[i], xyz1b[i], xyz1c[i]));
            float beta2 = acosf(gb2_comp / norm(xyz2a[i], xyz2b[i], xyz2c[i]));

            Complex temp;
            if(beta1<=phi && beta2<=phi){
                Complex vari((u[i]*l[index] + v[i]*m[index] + w[i]*(n[index]-1)), 0.0f);
                temp = Complex(B[index], 0) * complexExp((zero - I1) * two * CPI * vari);
                Viss[i] += temp;
            }
        }
        Viss[i+amount] = thrust::conj(Viss[i]);
    }
}


__global__ void viss_trans(Complex* Viss, float* w, int size, 
                Complex zero, Complex two, Complex CPI, Complex I1) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // gViss = gViss * exp(-1i * 2 * M_PI * gw)
        Complex cw(w[idx], 0.0);
        Viss[idx] = Viss[idx] * complexExp((zero-I1) * two * CPI * cw);
    }
}

            
__host__ float calculate_fa(const thrust::device_vector<float>& u,
                            const thrust::device_vector<float>& v,
                            const thrust::device_vector<float>& w) {
    float sum_v2 = thrust::transform_reduce(v.begin(), v.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());
    float sum_u_w = thrust::inner_product(u.begin(), u.end(), w.begin(), 0.0f);
    float sum_u_v = thrust::inner_product(u.begin(), u.end(), v.begin(), 0.0f);
    float sum_v_w = thrust::inner_product(v.begin(), v.end(), w.begin(), 0.0f);
    float sum_u2 = thrust::transform_reduce(u.begin(), u.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());

    return (sum_v2 * sum_u_w - sum_u_v * sum_v_w) / (sum_u2 * sum_v2 - sum_u_v * sum_u_v);
}


__host__ float calculate_fb(const thrust::device_vector<float>& u,
                            const thrust::device_vector<float>& v,
                            const thrust::device_vector<float>& w) {
    float sum_v2 = thrust::transform_reduce(v.begin(), v.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());
    float sum_u_w = thrust::inner_product(u.begin(), u.end(), w.begin(), 0.0f);
    float sum_u_v = thrust::inner_product(u.begin(), u.end(), v.begin(), 0.0f);
    float sum_v_w = thrust::inner_product(v.begin(), v.end(), w.begin(), 0.0f);
    float sum_u2 = thrust::transform_reduce(u.begin(), u.end(), thrust::square<float>(), 0.0f, thrust::plus<float>());

    return (sum_u2 * sum_v_w - sum_u_v * sum_u_w) / (sum_u2 * sum_v2 - sum_u_v * sum_u_v);
}


__global__ void meshgrid_kernel(float* ug, float* vg, int RES, float start, float step) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < RES * RES) {
        int i = idx / RES;
        int j = idx % RES;
        ug[idx] = start + i * step;
        vg[idx] = start + j * step;
    }
}


__global__ void round_divide_kernel(float* vec, float du, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vec[idx] = roundf(vec[idx] / du);
    }
}


__global__ void calculate_locg(const float* u, const float* v, int* locg, int n, int RES) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        locg[idx] = (u[idx] + (RES - 1) / 2) * RES + (v[idx] + (RES - 1) / 2) + 1;
    }
}



__global__ void computeC(
        int npix, float* ug, float* vg, Complex* Viss, 
        int* il, int* ulocg, int* locg, Complex* C, 
        float *xyz1a, float *xyz1b, float *xyz1c, 
        float *xyz2a, float *xyz2b, float *xyz2c,
        float* l, float* m, float* n, 
        int nulocg, float fa, float fb, float phi,
        int uvw_index, Complex two, Complex I1, Complex CPI)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < npix) {
        for (int q=0; q<nulocg; q++) {
            // ugu=ug(ulocg(q));  vgu=vg(ulocg(q));
            int loc = ulocg[q];
            float ugu = ug[loc-1];
            float vgu = vg[loc-1];

            // Find indices where il == q
            int count = 0;
            Complex Vissgu(0.0, 0.0);
            for (int i = 0; i < uvw_index; i++) {
                if (il[i] == q) {
                    Vissgu += Viss[i];
                    count++;
                }
            }

            // Compute acosbeta1 and beta1
            float acosbeta1[count];
            for (int i = 0; i < count; i++) {
                int locIdx = locViss[i];
                float dotProd = l[idx] * x1[locIdx] + m[idx] * y1[locIdx] + n[idx] * z1[locIdx];
                float norm = sqrtf(x1[locIdx] * x1[locIdx] + y1[locIdx] * y1[locIdx] + z1[locIdx] * z1[locIdx]);
                acosbeta1[i] = dotProd / norm;
                if (fabsf(acosbeta1[i]) > 1.0f) {
                    acosbeta1[i] = copysignf(1.0f, acosbeta1[i]);
                }
            }

            float beta1[100];
            for (int i = 0; i < locVissCount; i++) {
                beta1[i] = acosf(acosbeta1[i]);
            }


            Vissgu = Vissgu / Complex(count, 0.0);   // 一个数值
            
            Complex Vissp = Vissgu * complexExp(two * I1 * CPI * Complex((ugu * fa + vgu * fb) * n[idx], 0.0));

            // Update gC
            C[idx] += Vissp * complexExp(two * I1 * CPI * Complex(ugu * l[idx] + vgu * m[idx], 0.0));
        }
    }
}



int vissGen(float frequency) 
{   
    gettimeofday(&start, NULL);

    int nDevices;
    // 设置节点数量（gpu显卡数量）
    CHECK(cudaGetDeviceCount(&nDevices));
    // 设置并行区中的线程数
    omp_set_num_threads(nDevices);
    cout << "devices: " << nDevices << endl;

    cout << "frequency: " << frequency << endl;

    int days = 1;
    Complex I1(0.0, 1.0);
    Complex zero(0.0, 0.0);
    Complex one(1.0, 0.0);
    Complex two(2.0, 0.0);
    Complex CPI(M_PI, 0.0);
    cout << "days: " << days << endl;

    // 读取 B.txt, theta_heal.txt, phi_heal.txt 文件
    string address_B = "B.txt";
    string address_theta_heal = "theta_heal.txt";
    string address_phi_heal = "phi_heal.txt";
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

        // 将 B, theta_heal, phi_heal 数据从CPU搬到GPU上        
        thrust::device_vector<float> B(cB.begin(), cB.end());
        thrust::device_vector<float> theta_heal(ctheta_heal.begin(), ctheta_heal.end());
        thrust::device_vector<float> phi_heal(cphi_heal.begin(), cphi_heal.end());

        // 创建 l m n
        thrust::device_vector<float> l(npix), m(npix), n(npix);

        std::vector<float> cu(uvw_presize), cv(uvw_presize), cw(uvw_presize);
        thrust::device_vector<float> u(uvw_presize), v(uvw_presize), w(uvw_presize);

        std::vector<float> cxyz1a(uvw_presize), cxyz1b(uvw_presize), cxyz1c(uvw_presize);
        thrust::device_vector<float> xyz1a(uvw_presize), xyz1b(uvw_presize), xyz1c(uvw_presize);

        std::vector<float> cxyz2a(uvw_presize), cxyz2b(uvw_presize), cxyz2c(uvw_presize);
        thrust::device_vector<float> xyz2a(uvw_presize), xyz2b(uvw_presize), xyz2c(uvw_presize);

        std::vector<float> cbll(uvw_presize);
        thrust::device_vector<float> bll(uvw_presize);

        std::vector<Complex> cViss(uvw_presize);
        thrust::device_vector<Complex> Viss(uvw_presize);

        // 存储最终的计算结果
        thrust::device_vector<Complex> C(npix, zero);

        // 遍历所有开启的线程处理， 一个线程控制一个GPU 处理一个id*amount/total的块
        for (int p = tid; p < days; p += nDevices) {
            cout << "for loop: " << p+1 << endl;

            int uvw_index, xyz1_index, xyz2_index, bll_index; 
            #pragma omp critical
            {   
                // 读取 uvw
                string address_uvw = address + "uvw" + to_string(p+1) + "day1M.txt";
                cout << "address_uvw: " << address_uvw << endl;
                ifstream uvwFile(address_uvw);
                uvw_index = 0;
                float u_point, v_point, w_point;
                if (uvwFile.is_open()) {
                    uvwFile >> u_point >> v_point >> w_point; // 读取第一行，删除
                    while (uvwFile >> u_point >> v_point >> w_point) {
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

                // 读取Viss
                int viss_index;
                string address_viss = address + "Viss" + to_string(p+1) + "day1M.txt";
                cout << "address_viss: " << address_viss << endl;
                ifstream vissFile(address_viss);
                viss_index = 0;
                if (vissFile.is_open()) {
                    vissFile >> a_point >> b_point;
                    while (vissFile >> a_point >> b_point) {
                        cViss[viss_index].real(a_point);
                        cViss[viss_index].imag(b_point);
                        viss_index++;
                    }
                }
                cout << "viss_index: " << viss_index << endl;
                // 复制到GPU上
                thrust::copy(cViss.begin(), cViss.begin() + viss_index, Viss.begin());
            
            }

            // 计算可见度
            int amount = ceil(uvw_index/2);
            cout << "amount: " << amount << endl;
            int blockSize;
            int minGridSize; // 最小网格大小
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, healpix_moonback_pre, 0, 0);
            int gridSize = floor(npix + blockSize - 1) / blockSize;;  
            

            healpix_moonback_pre<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(theta_heal.data()), 
                thrust::raw_pointer_cast(phi_heal.data()),
                thrust::raw_pointer_cast(l.data()),
                thrust::raw_pointer_cast(m.data()),
                thrust::raw_pointer_cast(n.data()),
                thrust::raw_pointer_cast(B.data()),
                npix, s);
            CHECK(cudaDeviceSynchronize());  


            // thrust::device_vector<Complex> Viss(uvw_index);
            // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, healpix_moonback_viss, 0, 0);
            // gridSize = floor(amount + blockSize - 1) / blockSize;
            // cout << "Viss Computing, girdSize: " << gridSize << endl;
            // cout << "Viss Computing, blockSize: " << blockSize << endl;
            // printf("Viss Computing... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            // healpix_moonback_viss<<<gridSize, blockSize>>>(
            //     thrust::raw_pointer_cast(B.data()),
            //     thrust::raw_pointer_cast(Viss.data()),
            //     thrust::raw_pointer_cast(u.data()),
            //     thrust::raw_pointer_cast(v.data()),
            //     thrust::raw_pointer_cast(w.data()),
            //     thrust::raw_pointer_cast(xyz1a.data()),
            //     thrust::raw_pointer_cast(xyz1b.data()),
            //     thrust::raw_pointer_cast(xyz1c.data()),
            //     thrust::raw_pointer_cast(xyz2a.data()),
            //     thrust::raw_pointer_cast(xyz2b.data()),
            //     thrust::raw_pointer_cast(xyz2c.data()),
            //     thrust::raw_pointer_cast(l.data()),
            //     thrust::raw_pointer_cast(m.data()),
            //     thrust::raw_pointer_cast(n.data()),
            //     amount, npix, phi,
            //     zero, I1, two, CPI);
            // CHECK(cudaDeviceSynchronize());
            // cout << "Period " << p+1 << " Viss Computing Success!" << endl;
             

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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, viss_trans, 0, 0);
            gridSize = floor(uvw_index + blockSize - 1) / blockSize;
            viss_trans<<<gridSize, blockSize>>>(
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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, meshgrid_kernel, 0, 0);
            gridSize = floor(RES * RES + blockSize - 1) / blockSize;
            cout << "meshgrid, girdSize: " << gridSize << endl;
            cout << "meshgrid, blockSize: " << blockSize << endl;
            printf("meshgrid... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            meshgrid_kernel<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(ug.data()), 
                thrust::raw_pointer_cast(vg.data()), 
                RES, start, step);
            CHECK(cudaDeviceSynchronize());

            // 四舍五入 u 和 v
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, round_divide_kernel, 0, 0);
            gridSize = floor(u.size() + blockSize - 1) / blockSize;
            round_divide_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(u.data()), du, u.size());
            round_divide_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(v.data()), du, v.size());
            CHECK(cudaDeviceSynchronize());

            // 计算 
            thrust::device_vector<int> locg(uvw_index);
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, calculate_locg, 0, 0);
            gridSize = floor(uvw_index + blockSize - 1) / blockSize;
            cout << "locg, girdSize: " << gridSize << endl;
            cout << "locg, blockSize: " << blockSize << endl;
            printf("locg... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            calculate_locg<<<gridSize, blockSize>>>(
                thrust::raw_pointer_cast(u.data()), 
                thrust::raw_pointer_cast(v.data()), 
                thrust::raw_pointer_cast(locg.data()), 
                uvw_index, RES);
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

// -----------------------------------------------------------------------------------------------------------------------------------------------------
            // 计算 C
            int nulocg = ulocg.size();

            调用核函数计算C
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, computeC, 0, 0);
            gridSize = floor(npix + blockSize - 1) / blockSize;
            cout << "Compute C, girdSize: " << gridSize << endl;
            cout << "Compute C, blockSize: " << blockSize << endl;
            printf("Compute C... Here is gpu %d running process %d\n", omp_get_thread_num(), p+1);
            computeC<<<gridSize, blockSize>>>(
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
                nulocg, fa, fb,
                RES, npix, two, I1, CPI);
            cudaDeviceSynchronize();
            cout << "Period " << p+1 << " compute C success" << endl;

            string address_C = "wsblockage/C" + to_string(p+1) + "day1M.txt";
            writeToFile(C, address_C);
        }
    }
    
    gettimeofday(&finish, NULL);
    total_time = ((finish.tv_sec - start.tv_sec) * 1000000 + (finish.tv_usec - start.tv_usec)) / 1000000.0;
    cout << "total time: " << total_time << "s" << endl;
    return 0;
}


int main()
{
    vissGen(1e6);
}

