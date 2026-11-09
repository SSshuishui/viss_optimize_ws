#!/usr/bin/env bash
set -euo pipefail
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
nvcc_path="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
mkdir -p "$batch_root/bin"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLOOKUP_FIXED=0 -DATOMIC_TASKLIST=0 "$batch_root/scripts/replay_legacy_fastpath.cu" -o "$batch_root/bin/replay_legacy_fastpath_oldlookup"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLOOKUP_FIXED=1 -DATOMIC_TASKLIST=0 "$batch_root/scripts/replay_legacy_fastpath.cu" -o "$batch_root/bin/replay_legacy_fastpath_fixedlookup"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLOOKUP_FIXED=0 -DATOMIC_TASKLIST=1 "$batch_root/scripts/replay_legacy_fastpath.cu" -o "$batch_root/bin/replay_legacy_fastpath_atomic_oldlookup"
