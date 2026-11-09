#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/source/final_sts/scripts/run_one.sh"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiment_results/scaling_10m}"
: "${SKY_DIR:?Set SKY_DIR to the directory containing B_10M.bin}"

if [[ ! -x "$ROOT/source/final_sts/main_recon_online_mirror_tile_vissfast" ]]; then
  echo "Build source/final_sts before running this experiment." >&2
  exit 1
fi

for gpus in ${GPU_LIST:-"0 0,1 0,1,2,3"}; do
  label="g${gpus//,/}"
  out="$OUT_ROOT/$label"
  mkdir -p "$out"
  env SKY_DIR="$SKY_DIR" GPUS="$gpus" DAY=1 SEGS="${SEGS:-10}" \
    GRID_CELL_SCALE=1.0 OCC_MODE=2 WRITE_OUTPUT="${WRITE_OUTPUT:-0}" \
    PROFILE=1 OUT_DIR="$out" RUN_TAG="scaling_$label" \
    "$RUNNER" 2>&1 | tee "$out/run.log"
done

