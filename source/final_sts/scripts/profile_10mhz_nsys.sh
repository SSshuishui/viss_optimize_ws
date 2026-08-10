#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=${OUT_DIR:-$ROOT/results_profile_10m}
mkdir -p "$OUT_DIR"

# Build with NVTX labels. Set ARCH if the target GPU is not sm_89.
ENABLE_NVTX=1 "$ROOT/build.sh"

EXE="$ROOT/main_recon_online_mirror_tile_vissfast"
SKY_DIR=${SKY_DIR:?Set SKY_DIR}
GPUS=${GPUS:-0,1,2,3}
NSIDE=${NSIDE:-4096}
TIME_COUNT=${TIME_COUNT:-0}

nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=none \
  -o "$OUT_DIR/final_method_10m_day1" \
  "$EXE" \
    --btag=10M --nside="$NSIDE" --day_start=1 --day_count=1 \
    --segs=10 --grid_cell_scale=1.0 \
    --time_start=0 --time_count="$TIME_COUNT" \
    --blockage=1 --occ_mode=2 --sky_dir="$SKY_DIR" --out_dir="$OUT_DIR" \
    --gpus="$GPUS" --gen_gpu_index=0 --B_mode=bin --C_mode=bin \
    --profile=1 --write_output=0 --run_tag=profile10m

nsys stats "$OUT_DIR/final_method_10m_day1.nsys-rep" > "$OUT_DIR/nsys_stats.txt"
echo "Generated $OUT_DIR/final_method_10m_day1.nsys-rep and nsys_stats.txt"
