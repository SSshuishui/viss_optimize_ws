#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
EXE=${EXE:-$ROOT/main_recon_online_mirror_tile_vissfast}

BTAG=${BTAG:-10M}
NSIDE=${NSIDE:-4096}
DAY=${DAY:-1}
SEGS=${SEGS:-10}
GRID_CELL_SCALE=${GRID_CELL_SCALE:-1.0}
OCC_MODE=${OCC_MODE:-2}   # 0=exact group-wise, 2=representative median baseline
TIME_START=${TIME_START:-0}
TIME_COUNT=${TIME_COUNT:-0}
GPUS=${GPUS:-0,1,2,3}
GEN_GPU_INDEX=${GEN_GPU_INDEX:-0}
SKY_DIR=${SKY_DIR:?Set SKY_DIR to the directory containing B_<BTAG>.bin/txt}
OUT_DIR=${OUT_DIR:-$ROOT/out}
RUN_TAG=${RUN_TAG:-segs${SEGS}_grid${GRID_CELL_SCALE}_occ${OCC_MODE}}
WRITE_OUTPUT=${WRITE_OUTPUT:-1}
PROFILE=${PROFILE:-1}
B_MODE=${B_MODE:-bin}
C_MODE=${C_MODE:-bin}

mkdir -p "$OUT_DIR"
"$EXE" \
  --btag="$BTAG" --nside="$NSIDE" \
  --day_start="$DAY" --day_count=1 \
  --segs="$SEGS" --grid_cell_scale="$GRID_CELL_SCALE" \
  --time_start="$TIME_START" --time_count="$TIME_COUNT" \
  --blockage=1 --occ_mode="$OCC_MODE" \
  --sky_dir="$SKY_DIR" --out_dir="$OUT_DIR" \
  --gpus="$GPUS" --gen_gpu_index="$GEN_GPU_INDEX" \
  --B_mode="$B_MODE" --C_mode="$C_MODE" --orbit_seed=42 \
  --profile="$PROFILE" --write_output="$WRITE_OUTPUT" --run_tag="$RUN_TAG"
