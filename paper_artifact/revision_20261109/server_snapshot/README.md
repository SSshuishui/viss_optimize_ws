# Server source snapshot, 9 November 2026

These eight CUDA source/header files were retrieved directly from the four-RTX-4090 evaluation server. SHA-256 identities are in SOURCE_IDENTITY.json. They retain the current server working-source state. The separately frozen implementations under agent_exchange and source identify the published timing and numerical batches. A current source file is not retroactively assigned as the producer of an older image.

Build each directory with CUDA 12.4 and C++17, targeting the installed GPU. For example, in mirror_tile_vissfast run `nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp main_recon_online_mirror_tile_vissfast.cu -o main_sts`. Pass sky input/output paths and GPU IDs explicitly using `--help`.
