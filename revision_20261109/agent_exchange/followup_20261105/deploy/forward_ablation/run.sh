#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo 'Usage: bash run.sh GPU-UUID SKY_FP32_BIN NEW_RUN_DIRECTORY' >&2
  exit 64
fi
gpu_uuid="$1"
sky_file="$2"
run_dir="$3"
task_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$gpu_uuid" in GPU-*) ;; *) echo 'Pass the coordinator-assigned GPU UUID.' >&2; exit 64;; esac
[ -x "$task_dir/build/forward_ablation" ]
(
  cd "$task_dir"
  sha256sum --quiet -c build/BUILD_SHA256SUMS
)
mkdir "$run_dir"
expected_sky_sha='277b6cb7299f2e8e54cc1758a045e7589686e28dc79a1df2f56aba5fc2d2d5ac'
printf '%s  %s\n' "$expected_sky_sha" "$sky_file" > "$run_dir/INPUT_SHA256SUMS"
sha256sum -c "$run_dir/INPUT_SHA256SUMS" > "$run_dir/input_check.txt"
cp "$task_dir/build/BUILD_SHA256SUMS" "$run_dir/BUILD_SHA256SUMS"
cp "$task_dir/build/kernel_change.diff" "$run_dir/kernel_change.diff"
cp "$task_dir/build/build_command.txt" "$run_dir/build_command.txt"
cp "$task_dir/build/nvcc_version.txt" "$run_dir/nvcc_version.txt"
exec 9>"$task_dir/run.lock"
flock -n 9 || { echo 'A run for this package is already active.' >&2; exit 10; }
nvidia-smi -i "$gpu_uuid" --query-gpu=index,uuid,name,driver_version,memory.free,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv > "$run_dir/gpu_preflight.csv"
nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_gpu_memory \
  --format=csv,noheader > "$run_dir/compute_preflight.csv"
if [ -s "$run_dir/compute_preflight.csv" ]; then
  echo 'Assigned GPU has an active compute process.' >&2
  exit 10
fi
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'gpu_uuid=%s\nwall_limit_s=150\n' "$gpu_uuid"
  uname -sr
} > "$run_dir/launch.txt"
set +e
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu_uuid" OMP_NUM_THREADS=4 \
  timeout --signal=TERM --kill-after=10s 150s "$task_dir/build/forward_ablation" "$sky_file" "$run_dir" \
  > "$run_dir/process.log" 2>&1
run_rc=$?
set -e
printf '%d\n' "$run_rc" > "$run_dir/exit_code.txt"
nvidia-smi -i "$gpu_uuid" --query-gpu=index,uuid,name,driver_version,memory.free,temperature.gpu,clocks.sm,clocks.mem \
  --format=csv > "$run_dir/gpu_postflight.csv"
if [ "$run_rc" -eq 0 ] || [ "$run_rc" -eq 2 ]; then
  python3 "$task_dir/summarize.py" "$run_dir"
fi
(
  cd "$run_dir"
  find . -maxdepth 1 -type f ! -name OUTPUT_SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > OUTPUT_SHA256SUMS
)
exit "$run_rc"
