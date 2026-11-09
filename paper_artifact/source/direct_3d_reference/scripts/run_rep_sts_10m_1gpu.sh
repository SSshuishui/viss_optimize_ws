#!/usr/bin/env bash
set -euo pipefail

REFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FINAL_METHOD_DIR="${FINAL_METHOD_DIR:-$REFERENCE_DIR/../final_sts}"
: "${SKY_DIR:?Set SKY_DIR to the directory containing B_10M.bin}"
OUT_DIR="${OUT_DIR:-$FINAL_METHOD_DIR/out_rep_sts_10m_1gpu}"
GPUS="${GPUS:-0}"

if [[ ! -x "$FINAL_METHOD_DIR/scripts/run_one.sh" ]]; then
  echo "ERROR: $FINAL_METHOD_DIR/scripts/run_one.sh not found or not executable" >&2
  exit 1
fi

cd "$FINAL_METHOD_DIR"
mkdir -p "$OUT_DIR"

exec env \
  SKY_DIR="$SKY_DIR" \
  GPUS="$GPUS" \
  BTAG=10M \
  NSIDE=4096 \
  DAY="${DAY:-1}" \
  SEGS=10 \
  GRID_CELL_SCALE=1.0 \
  OCC_MODE=2 \
  TIME_START=0 \
  TIME_COUNT=0 \
  WRITE_OUTPUT=1 \
  PROFILE=1 \
  OUT_DIR="$OUT_DIR" \
  RUN_TAG=rep_sts_1gpu \
  "$FINAL_METHOD_DIR/scripts/run_one.sh"
