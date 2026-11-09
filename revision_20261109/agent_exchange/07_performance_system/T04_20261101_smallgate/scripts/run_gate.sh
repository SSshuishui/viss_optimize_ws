#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/usr/local/cuda-12.4/bin:$PATH"
mkdir -p "$TASK_DIR/logs" "$TASK_DIR/metrics"
exec 9>"$TASK_DIR/gate.lock"
flock -n 9 || exit 9
# Physical GPU2 was released for this bounded experiment by the coordinating agent.
# Check active processes, not trailing rolling utilization after a prior run.
nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_gpu_memory --format=csv > "$TASK_DIR/metrics/gpu2_preflight.csv"
if [ "$(wc -l < "$TASK_DIR/metrics/gpu2_preflight.csv")" -gt 1 ]; then
  echo 'GPU2 has another compute process; refusing launch.' >&2; exit 10
fi
export CUDA_VISIBLE_DEVICES=2
set +e
timeout 120 /usr/local/cuda-12.4/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$TASK_DIR/bin/t04_smallgate" --shared-arrays=4 --output="$TASK_DIR/metrics/negative_shared4.csv" > "$TASK_DIR/logs/sanitizer_shared4.log" 2>&1
negative_rc=$?
timeout 120 /usr/local/cuda-12.4/bin/compute-sanitizer --tool memcheck --error-exitcode 99 "$TASK_DIR/bin/t04_smallgate" --shared-arrays=5 --output="$TASK_DIR/metrics/corrected_shared5.csv" > "$TASK_DIR/logs/sanitizer_shared5.log" 2>&1
positive_rc=$?
set -e
printf 'negative_shared4_rc=%s\ncorrected_shared5_rc=%s\n' "$negative_rc" "$positive_rc" > "$TASK_DIR/metrics/gate_exit_codes.txt"
[ "$positive_rc" -eq 0 ] || exit "$positive_rc"
python3 "$TASK_DIR/scripts/analyze_smallgate.py" "$TASK_DIR/metrics/corrected_shared5.csv" "$TASK_DIR/metrics/smallgate_analysis.json" > "$TASK_DIR/logs/analysis.log"
du -sb "$TASK_DIR" > "$TASK_DIR/metrics/directory_size.txt"
cat "$TASK_DIR/metrics/gate_exit_codes.txt" "$TASK_DIR/logs/analysis.log"
