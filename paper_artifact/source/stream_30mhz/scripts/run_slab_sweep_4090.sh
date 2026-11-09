#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${SKY_DIR:?Set SKY_DIR to the directory containing B_30M.bin or B.bin}"

GPUS=${GPUS:-0,1,2,3}
DAY=${DAY:-1}
SEGS=${SEGS:-10}
SLAB_LIST=${SLAB_LIST:-"33554432 67108864 134217728 268435456"}
REPEATS=${REPEATS:-1}
OUT_ROOT=${OUT_ROOT:-$ROOT_DIR/results_30m_4090_slab_sweep}
WRITE_OUTPUT=${WRITE_OUTPUT:-0}
SUMMARY_CSV=${SUMMARY_CSV:-$OUT_ROOT/slab_sweep_summary.csv}

mkdir -p "$OUT_ROOT"
rm -f "$SUMMARY_CSV"

{
  echo "date=$(date --iso-8601=seconds 2>/dev/null || date)"
  echo "host=$(hostname)"
  echo "gpus=$GPUS"
  echo "day=$DAY"
  echo "segs=$SEGS"
  echo "slab_list=$SLAB_LIST"
  echo "repeats=$REPEATS"
  echo "write_output=$WRITE_OUTPUT"
} > "$OUT_ROOT/sweep_config.txt"

for slab in $SLAB_LIST; do
  for ((r=1; r<=REPEATS; ++r)); do
    echo "===== 30 MHz slab sweep: slab=$slab repeat=$r/$REPEATS day=$DAY ====="
    SKY_DIR="$SKY_DIR" \
    GPUS="$GPUS" \
    DAY="$DAY" \
    SEGS="$SEGS" \
    SLAB_PIX="$slab" \
    OUT_ROOT="$OUT_ROOT" \
    WRITE_OUTPUT="$WRITE_OUTPUT" \
    RUN_TAG="slab_${slab}_r${r}" \
    SUMMARY_CSV="$SUMMARY_CSV" \
    "$ROOT_DIR/scripts/run_30m_one_day_4090.sh"
  done
done

python3 "$ROOT_DIR/tools/summarize_slab_sweep.py" \
  --input "$SUMMARY_CSV" \
  --csv-out "$OUT_ROOT/slab_sensitivity_table.csv" \
  --md-out "$OUT_ROOT/slab_sensitivity_table.md"

echo "[OK] Raw summary: $SUMMARY_CSV"
echo "[OK] Table: $OUT_ROOT/slab_sensitivity_table.csv"
echo "[OK] Markdown: $OUT_ROOT/slab_sensitivity_table.md"
