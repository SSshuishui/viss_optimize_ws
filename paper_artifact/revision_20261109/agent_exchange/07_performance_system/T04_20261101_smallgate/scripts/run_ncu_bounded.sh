#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRODUCTION=/data/zhaox/research/viss_optimize_ws/system_viss_recon_3d_dcf_controlled/final_method_sensitivity_profile/main_recon_online_mirror_tile_vissfast
SKY=/data/zhaox/research/viss_optimize_ws/earth_10Mhz
METRICS=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread
export PATH="/usr/local/cuda-12.4/bin:$PATH"
mkdir -p "$TASK_DIR/logs" "$TASK_DIR/metrics" "$TASK_DIR/ncu_empty_output"
sha256sum "$PRODUCTION" > "$TASK_DIR/metrics/ncu_production_binary_sha256.txt"
for stage in forward representative_imaging; do
  if [ "$stage" = forward ]; then kernel=regex:viss_partial_all_halfsym_tilecone; else kernel=regex:recon_seg_blockage_half_rep; fi
  # A single production-size first segment, no image output. Kill target after
  # the one matching kernel is profiled, so this cannot replay an entire day.
  set +e
  timeout -k 10 120 /usr/local/cuda-12.4/bin/ncu --kernel-name "$kernel" --launch-count 1 --kill yes \
    --metrics "$METRICS" --csv --log-file "$TASK_DIR/logs/ncu_${stage}.csv" \
    "$PRODUCTION" --btag=10M --nside=4096 --day_start=1 --day_count=1 \
    --segs=10 --grid_cell_scale=1.0 \
    --blockage=1 --occ_mode=2 --gpus=2 --gen_gpu_index=0 \
    --sky_dir="$SKY" --B_mode=bin --C_mode=bin --write_output=0 \
    --out_dir="$TASK_DIR/ncu_empty_output/" \
    > "$TASK_DIR/logs/ncu_${stage}_application.log" 2>&1
  rc=$?
  set -e
  printf '%s,%s\n' "$stage" "$rc" >> "$TASK_DIR/metrics/ncu_exit_codes.csv"
  if grep -qE 'ERR_NVGPUCTRPERM|ERR_NVCMP_PERM' "$TASK_DIR/logs/ncu_${stage}.csv" "$TASK_DIR/logs/ncu_${stage}_application.log"; then exit 0; fi
done
du -sb "$TASK_DIR" > "$TASK_DIR/metrics/directory_size_after_ncu.txt"
cat "$TASK_DIR/metrics/ncu_exit_codes.csv"
