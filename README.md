# viss_optimize_ws (二维短时叠加算法)

可见度计算使用一半共轭 + 反演阶段使用基线分段，确保结果正确，并且计算速度提升 \
同时基线部分采用在线生成 \
$\phi$ 和 $\theta$ 也采用在线生成，同时用它生成lmn，常驻内存，并在分段的时候使用对应段的部分


## 文件结构
```text
├── main_recon_online.cu
├── common.hpp
├── orbit_online.cuh
├── viss_recon_kernel.cuh
└── README.md

main_recon_online.cu：主程序入口，负责参数解析、GPU 初始化、按段调度、在线基线生成、Viss 计算与图像反演。

common.hpp：公共头文件，包含基础库引用、通用宏、参数/文件/目录工具函数及公共数据结构定义。

orbit_online.cuh：在线轨道与基线生成模块，负责生成 segment 级 uvw / xyza / xyzb，并提供相关校验与辅助函数。

viss_recon_kernel.cuh：Viss 计算与重建核心 CUDA 核函数，包含半量可见度计算、相位恢复、共轭补全和按段二维反演。

README.md：项目说明文档。
```

## 天图反演模拟
### 编译
```bash
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp main_recon_online.cu -o main_recon_online
```
### 运行
```bash
bash main_recon_online.sh
```

## 测试对比一致性
由于之前是加载的 $\phi$(phi_heal.txt) 和 $\theta$(theta_heal.txt) ，在精度上有一些差别，因此在原来分段的基础上测试了在线生成 lmn，结果显示相同 \
如果需要没有区别，选择高精度的 double 类型，应该能完全一致
### 编译
```bash
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -fopenmp recon_seg_clip_viss_conj_viss_seg_online.cu -o recon_seg_clip_viss_conj_viss_seg_online
```
### 运行
```bash
bash recon_seg_clip_viss_conj_viss_seg_online.sh
```