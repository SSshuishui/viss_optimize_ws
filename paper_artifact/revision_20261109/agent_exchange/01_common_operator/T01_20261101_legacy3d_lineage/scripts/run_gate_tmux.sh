#!/usr/bin/env bash
set -u
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
python3 -u "$batch_root/scripts/run_short_gate.py" > "$batch_root/execution.log" 2>&1
gate_rc=$?
printf '%s\n' "$gate_rc" > "$batch_root/execution.exitcode"
tmux wait-for -S legacy3d_gate_done_20261101
exit "$gate_rc"
