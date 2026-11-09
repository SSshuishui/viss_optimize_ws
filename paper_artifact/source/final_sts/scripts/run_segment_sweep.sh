#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SEGS_LIST=${SEGS_LIST:-"5 10 20 40"}
GRID_CELL_SCALE=${GRID_CELL_SCALE:-1.0}
RUN_EXACT=${RUN_EXACT:-1}
OUT_ROOT=${OUT_ROOT:-$ROOT/results_segment_sweep}
mkdir -p "$OUT_ROOT"

for segs in $SEGS_LIST; do
  for mode in rep exact; do
    if [[ "$mode" == "exact" && "$RUN_EXACT" != "1" ]]; then continue; fi
    if [[ "$mode" == "rep" ]]; then occ=2; else occ=0; fi
    tag="segment_s${segs}_${mode}"
    echo "===== $tag ====="
    SEGS="$segs" GRID_CELL_SCALE="$GRID_CELL_SCALE" OCC_MODE="$occ" \
      RUN_TAG="$tag" OUT_DIR="$OUT_ROOT/$tag" \
      "$ROOT/scripts/run_one.sh"
  done

done

echo "Profile summaries:"
find "$OUT_ROOT" -name 'profile_*.csv' -print0 | xargs -0 -r "$ROOT/tools/summarize_profile.py" --out "$OUT_ROOT/profile_summary.csv"
