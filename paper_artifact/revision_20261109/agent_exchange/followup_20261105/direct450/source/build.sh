#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 )); then
  echo "Usage: CUDA_ARCH=sm_75 NVCC=/path/to/nvcc $0 OUTPUT [extra nvcc flags]" >&2
  exit 2
fi
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
output=$1
shift
arch=${CUDA_ARCH:-}
case "$arch" in
  sm_75|sm_86|sm_89) ;;
  *) echo "Set CUDA_ARCH to sm_75, sm_86, or sm_89" >&2; exit 2 ;;
esac
(
  cd "$source_dir"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum -c FROZEN_SHA256SUMS
  else
    shasum -a 256 -c FROZEN_SHA256SUMS
  fi
)
"${NVCC:-nvcc}" -O3 --use_fast_math -lineinfo -std=c++17 -arch="$arch" \
  -Xcompiler -fopenmp "$@" "$source_dir/direct_daily.cu" -o "$output" -lcrypto
