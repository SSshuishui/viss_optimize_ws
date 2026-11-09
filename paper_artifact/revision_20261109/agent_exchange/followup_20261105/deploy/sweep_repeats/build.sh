#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")" && pwd)"
exchange_root="${1:?Usage: bash build.sh /absolute/agent_exchange [nvcc-path]}"
nvcc="${2:-/usr/local/cuda-12.4/bin/nvcc}"
host_cxx="${CUDA_HOST_CXX:-/usr/bin/g++-11}"
sts_root="$exchange_root/01_common_operator/T01_20261101_production_replay"

python3 "$task_dir/run_sweep_repeats.py" --exchange-root "$exchange_root" --verify-small-inputs
mkdir "$task_dir/build"
{
    "$nvcc" --version
    "$host_cxx" --version
    set -x
    "$nvcc" -ccbin "$host_cxx" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp \
        -I "$sts_root/source" -I "$sts_root/scripts" "$task_dir/replay_sts_sweep.cu" -lcrypto \
        -o "$task_dir/build/replay_sts_sweep"
} 2>&1 | tee "$task_dir/build/build.log"
