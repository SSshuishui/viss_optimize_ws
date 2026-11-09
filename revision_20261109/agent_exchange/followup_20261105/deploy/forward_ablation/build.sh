#!/usr/bin/env bash
set -euo pipefail
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$task_dir"
sha256sum -c FROZEN_SHA256SUMS
python3 make_point_kernel.py
nvcc_path="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
host_cxx="${CUDA_HOST_CXX:-/usr/bin/g++-11}"
"$nvcc_path" --version > build/nvcc_version.txt
"$host_cxx" --version > build/host_cxx_version.txt
cmd=("$nvcc_path" -ccbin "$host_cxx" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp
  -I frozen -I build forward_ablation.cu -o build/forward_ablation)
printf '%q ' "${cmd[@]}" > build/build_command.txt
printf '\n' >> build/build_command.txt
"${cmd[@]}"
sha256sum forward_ablation.cu make_point_kernel.py build.sh run.sh summarize.py FROZEN_SHA256SUMS frozen/* \
  build/point_only.cuh build/kernel_change.diff build/nvcc_version.txt build/host_cxx_version.txt build/build_command.txt \
  build/forward_ablation > build/BUILD_SHA256SUMS
