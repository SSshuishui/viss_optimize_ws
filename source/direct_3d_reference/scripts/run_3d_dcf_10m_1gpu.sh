#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${SKY_DIR:?Set SKY_DIR to the directory containing B_10M.bin}"
OUT_DIR="${OUT_DIR:-$ROOT/out_3d_dcf_10m_1gpu}"
GPUS="${GPUS:-0}"
SEGS="${SEGS:-10}"
DAY_START="${DAY_START:-1}"
DAY_COUNT="${DAY_COUNT:-1}"
ORBIT_SEED="${ORBIT_SEED:-42}"
VISS_TILE_PIX="${VISS_TILE_PIX:-256}"
RUN_TAG="${RUN_TAG:-3d_dcf_1gpu}"
DCF_BIN="${DCF_BIN:-$ROOT/dcf_mb_10M_days450_seed42.bin}"

mkdir -p "$OUT_DIR"

exec "$ROOT/main_3d_vissfast" \
  --btag=10M \
  --nside=4096 \
  --day_start="$DAY_START" \
  --day_count="$DAY_COUNT" \
  --segs="$SEGS" \
  --sky_dir="$SKY_DIR" \
  --out_dir="$OUT_DIR" \
  --gpus="$GPUS" \
  --gen_gpu_index=0 \
  --reducer_gpu_index=0 \
  --B_mode=bin \
  --C_mode=bin \
  --orbit_seed="$ORBIT_SEED" \
  --dcf_bin="$DCF_BIN" \
  --viss_tile_pix="$VISS_TILE_PIX" \
  --write_viss=0 \
  --write_baseline_txt=0 \
  --run_tag="$RUN_TAG"
