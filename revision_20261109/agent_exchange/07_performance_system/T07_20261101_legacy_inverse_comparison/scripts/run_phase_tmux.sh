#!/usr/bin/env bash
set -u
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
phase="${1:?gates or formal}"
python3 -u "$batch_root/scripts/run_comparison.py" --phase "$phase" > "$batch_root/${phase}_execution.log" 2>&1
phase_rc=$?
printf '%s\n' "$phase_rc" > "$batch_root/${phase}_execution.exitcode"
if [ "$phase" = gates ]; then
  tmux wait-for -S legacy_inverse_gates_done_20261101
else
  tmux wait-for -S legacy_inverse_formal_done_20261101
fi
exit "$phase_rc"
