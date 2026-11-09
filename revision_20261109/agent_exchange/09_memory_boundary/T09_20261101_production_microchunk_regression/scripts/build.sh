#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$root/bin" "$root/logs"
python3 "$root/scripts/prepare_source.py" > "$root/logs/prepare_source.log"
"${NVCC:-/usr/local/cuda-12.4/bin/nvcc}" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -I"$root/source" "$root/source/main_production_regression.cu" -lcrypto -o "$root/bin/production_microchunk_regression" > "$root/logs/build.log" 2>&1
sha256sum "$root/bin/production_microchunk_regression" > "$root/logs/binary.sha256"
