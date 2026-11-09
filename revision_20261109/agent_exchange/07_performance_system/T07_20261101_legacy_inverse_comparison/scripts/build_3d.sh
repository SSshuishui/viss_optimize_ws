#!/usr/bin/env bash
set -euo pipefail
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
nvcc_path="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
mkdir -p "$batch_root/bin"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLEGACY_FASTPATH=0 "$batch_root/scripts/replay_3d_full.cu" -lcrypto -o "$batch_root/bin/replay_controlled_fixed_full"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLEGACY_FASTPATH=1 "$batch_root/scripts/replay_3d_full.cu" -lcrypto -o "$batch_root/bin/replay_legacy_combined_full"
