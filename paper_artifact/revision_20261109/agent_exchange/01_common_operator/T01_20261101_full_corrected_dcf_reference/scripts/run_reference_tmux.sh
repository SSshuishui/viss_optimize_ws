#!/usr/bin/env bash
set -u
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -u "$batch_root/scripts/run_reference.py" --execute > "$batch_root/execution.log" 2>&1
run_rc=$?
printf '%s\n' "$run_rc" > "$batch_root/execution.exitcode"
tmux wait-for -S full_corrected_dcf_reference_done_20261101
exit "$run_rc"
