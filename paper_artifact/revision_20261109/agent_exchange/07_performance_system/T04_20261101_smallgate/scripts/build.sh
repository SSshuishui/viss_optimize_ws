#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$TASK_DIR/bin" "$TASK_DIR/logs"
/usr/local/cuda-12.4/bin/nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -I "$TASK_DIR/frozen" "$TASK_DIR/scripts/t04_smallgate.cu" -o "$TASK_DIR/bin/t04_smallgate"
sha256sum "$TASK_DIR/bin/t04_smallgate" "$TASK_DIR/scripts/t04_smallgate.cu" "$TASK_DIR/frozen/"* > "$TASK_DIR/metrics/build_sha256.txt"
