#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$ROOT/source/final_sts/scripts/run_one.sh"
COMPARE="$ROOT/source/final_sts/tools/compare_maps.py"
OUT_ROOT="${OUT_ROOT:-$ROOT/experiment_results/five_day_paired_10m}"
: "${SKY_DIR:?Set SKY_DIR to the directory containing B_10M.bin}"

if [[ ! -x "$ROOT/source/final_sts/main_recon_online_mirror_tile_vissfast" ]]; then
  echo "Build source/final_sts before running this experiment." >&2
  exit 1
fi

write_output="${WRITE_OUTPUT:-1}"
for day in ${DAYS:-"1 113 225 337 450"}; do
  for mode in exact representative; do
    if [[ "$mode" == exact ]]; then occ_mode=0; else occ_mode=2; fi
    out="$OUT_ROOT/day${day}_${mode}"
    tag="day${day}_${mode}"
    mkdir -p "$out"
    env SKY_DIR="$SKY_DIR" GPUS="${GPUS:-0,1,2,3}" DAY="$day" \
      SEGS="${SEGS:-10}" GRID_CELL_SCALE=1.0 OCC_MODE="$occ_mode" \
      WRITE_OUTPUT="$write_output" PROFILE=1 OUT_DIR="$out" RUN_TAG="$tag" \
      "$RUNNER" 2>&1 | tee "$out/run.log"
  done

  if [[ "$write_output" == 1 ]]; then
    python3 "$COMPARE" \
      --reference "$OUT_ROOT/day${day}_exact/C${day}day10M_day${day}_exact.bin" \
      --candidate "$OUT_ROOT/day${day}_representative/C${day}day10M_day${day}_representative.bin" \
      --label "day${day}_representative_vs_exact" \
      --json-out "$OUT_ROOT/day${day}_metrics.json" \
      --csv-out "$OUT_ROOT/day${day}_metrics.csv"
  fi
done
