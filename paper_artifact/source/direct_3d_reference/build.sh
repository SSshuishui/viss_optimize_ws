#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_89}"

"$NVCC" -O3 --use_fast_math -lineinfo -std=c++17 \
  -arch="$ARCH" -Xcompiler -fopenmp \
  main_3d_vissfast.cu -o main_3d_vissfast

echo "Built: $SCRIPT_DIR/main_3d_vissfast"
