#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
GRID_LIST=${GRID_LIST:-"0.75 1.0 1.25 1.5"}
SEGS=${SEGS:-10}
RUN_EXACT=${RUN_EXACT:-1}
OUT_ROOT=${OUT_ROOT:-$ROOT/results_grid_sweep}
mkdir -p "$OUT_ROOT"

for scale in $GRID_LIST; do
  safe_scale=${scale//./p}
  for mode in rep exact; do
    if [[ "$mode" == "exact" && "$RUN_EXACT" != "1" ]]; then continue; fi
    if [[ "$mode" == "rep" ]]; then occ=2; else occ=0; fi
    tag="grid_g${safe_scale}_${mode}"
    echo "===== $tag ====="
    GRID_CELL_SCALE="$scale" SEGS="$SEGS" OCC_MODE="$occ" \
      RUN_TAG="$tag" OUT_DIR="$OUT_ROOT/$tag" \
      "$ROOT/scripts/run_one.sh"
  done

done

echo "Profile summaries:"
find "$OUT_ROOT" -name 'profile_*.csv' -print0 | xargs -0 -r "$ROOT/tools/summarize_profile.py" --out "$OUT_ROOT/profile_summary.csv"
