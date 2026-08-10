#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT"

NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_89}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
NVTX_DEF=""
NVTX_LIB=""
if [[ "${ENABLE_NVTX:-0}" == "1" ]]; then
  NVTX_DEF="-DENABLE_NVTX"
  NVTX_LIB="-lnvToolsExt"
fi

$NVCC -O3 --use_fast_math -lineinfo -std=c++17 -arch="$ARCH" \
  -Xcompiler -fopenmp $EXTRA_FLAGS $NVTX_DEF \
  main_recon_online_mirror_tile_vissfast.cu \
  -o main_recon_online_mirror_tile_vissfast $NVTX_LIB
