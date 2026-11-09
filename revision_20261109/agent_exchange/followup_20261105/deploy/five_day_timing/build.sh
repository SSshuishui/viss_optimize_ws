#!/usr/bin/env bash
set -euo pipefail

task_dir="$(cd "$(dirname "$0")" && pwd)"
exchange_root="${1:?Usage: bash build.sh /absolute/agent_exchange [nvcc-path]}"
nvcc="${2:-/usr/local/cuda-12.4/bin/nvcc}"
host_cxx="${CUDA_HOST_CXX:-/usr/bin/g++-11}"
build_dir="$task_dir/build"
legacy_headers="$exchange_root/07_performance_system/T07_20261101_legacy_inverse_comparison/source/legacy_combined"
sts_root="$exchange_root/01_common_operator/T01_20261101_production_replay"

python3 "$task_dir/run_five_day_timing.py" --exchange-root "$exchange_root" --verify-headers
if [[ -e "$build_dir" ]]; then
    printf 'Build directory already exists: %s\n' "$build_dir" >&2
    exit 2
fi
mkdir "$build_dir"
common_flags=(-ccbin "$host_cxx" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp)
{
    "$nvcc" --version
    "$host_cxx" --version
    set -x
    "$nvcc" "${common_flags[@]}" -DLEGACY_FASTPATH=1 -I "$legacy_headers" \
        "$task_dir/replay_3d_full.cu" -lcrypto -o "$build_dir/replay_legacy_combined_full"
    "$nvcc" "${common_flags[@]}" -I "$sts_root/source" -I "$sts_root/scripts" \
        "$task_dir/replay_sts_full.cu" -lcrypto -o "$build_dir/replay_sts_full"
} 2>&1 | tee "$build_dir/build.log"
