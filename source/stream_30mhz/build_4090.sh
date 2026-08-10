#!/usr/bin/env bash
set -euo pipefail

NVCC=${NVCC:-nvcc}
OUT=${OUT:-main_recon_online_mirror_tile_vissfast_streamonly}

if ! command -v "$NVCC" >/dev/null 2>&1; then
  echo "[ERROR] nvcc not found. Load the CUDA environment first." >&2
  exit 1
fi

"$NVCC" \
  -O3 \
  --use_fast_math \
  -lineinfo \
  -std=c++17 \
  -Xcompiler -fopenmp \
  -arch=sm_89 \
  main_recon_online_mirror_tile_vissfast_streamonly.cu \
  -o "$OUT"

echo "[OK] Built $OUT for RTX 4090 (sm_89)."
