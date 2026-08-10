#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${BIN:-$ROOT_DIR/main_recon_online_mirror_tile_vissfast_streamonly}

: "${SKY_DIR:?Set SKY_DIR to the directory containing B_30M.bin or B.bin}"

GPUS=${GPUS:-0,1,2,3}
DAY=${DAY:-1}
SEGS=${SEGS:-10}
SLAB_PIX=${SLAB_PIX:-134217728}
OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/results_30m_4090}
WRITE_OUTPUT=${WRITE_OUTPUT:-0}
B_MODE=${B_MODE:-bin}
C_MODE=${C_MODE:-bin}
ORBIT_SEED=${ORBIT_SEED:-42}
VISS_TILE_PIX=${VISS_TILE_PIX:-256}
RUN_TAG=${RUN_TAG:-slab_${SLAB_PIX}}
SUMMARY_CSV=${SUMMARY_CSV:-$OUT_ROOT/slab_sweep_summary.csv}

IFS=',' read -r -a GPU_ARRAY <<< "$GPUS"
if [[ ${#GPU_ARRAY[@]} -ne 4 ]]; then
  echo "[ERROR] This script is for 4 GPUs; GPUS=$GPUS has ${#GPU_ARRAY[@]} entries." >&2
  exit 1
fi

if [[ ! -x "$BIN" ]]; then
  echo "[ERROR] Binary not found: $BIN" >&2
  echo "Run ./build_4090.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT" "$(dirname "$SUMMARY_CSV")"
RUN_DIR="$OUT_ROOT/day${DAY}_${RUN_TAG}"
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/run.log"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] Selected physical GPUs: $GPUS"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$RUN_DIR/gpu_info.txt"
  selected_names=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' -v ids="$GPUS" '
    BEGIN { n=split(ids,a,","); for(i=1;i<=n;i++) wanted[a[i]]=1 }
    wanted[$1] { print $2 }
  ')
  if [[ -n "$selected_names" ]] && grep -qv 'RTX 4090' <<< "$selected_names"; then
    echo "[WARNING] At least one selected GPU is not reported as RTX 4090:" >&2
    echo "$selected_names" >&2
  fi
fi

CMD=(
  "$BIN"
  --btag=30M
  --nside=16384
  --day_start="$DAY"
  --day_count=1
  --segs="$SEGS"
  --blockage=1
  --occ_mode=2
  --sky_dir="$SKY_DIR"
  --out_dir="$RUN_DIR"
  --gpus="$GPUS"
  --gen_gpu_index=0
  --B_mode="$B_MODE"
  --C_mode="$C_MODE"
  --orbit_seed="$ORBIT_SEED"
  --viss_tile_pix="$VISS_TILE_PIX"
  --sky_slab_pix="$SLAB_PIX"
  --write_viss=0
  --write_baseline_txt=0
  --write_output="$WRITE_OUTPUT"
  --run_tag="$RUN_TAG"
  --summary_csv="$SUMMARY_CSV"
  --print_stats=1
)

printf '[CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$LOG_FILE"
