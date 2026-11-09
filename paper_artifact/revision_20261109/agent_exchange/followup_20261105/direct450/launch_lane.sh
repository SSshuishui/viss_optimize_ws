#!/usr/bin/env bash
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
lane=${1:?lane id required}
uuid=${2:?GPU UUID required}
dates=${3:?assigned dates required}
[[ $lane =~ ^(a6000|2080ti)-[0-9]$ ]]
cd "$base"
mkdir -p lanes console
args=(--lane-root "$base/lanes/$lane" --gpu-uuid "$uuid" --dates "$dates"
      --forward-bin "$base/bin/forward_uuid" --direct-bin "$base/bin/direct_daily"
      --sky-file "$base/inputs/earth_10Mhz/B_10M.bin"
      --dcf-bin "$base/inputs/dcf_mb_10M_days450_seed42.bin"
      --build-receipt "$base/build_receipt.txt" --omp-threads 4)

python3 - "$base" <<'PY'
import hashlib, json, sys
from pathlib import Path
p = Path(sys.argv[1])
gate = p / "portability_gate"
for report, binary, key in (("forward_gate.json", "forward_uuid", "forward_sha256"),
                            ("direct_gate.json", "direct_daily", "direct_sha256")):
    data = json.loads((gate / report).read_text())
    if data.get("passed") is not True or data[key] != hashlib.sha256((p / "bin" / binary).read_bytes()).hexdigest():
        raise SystemExit(f"Missing passing portability gate for {binary}")
PY

python3 -B -u runner/run_lane.py --phase preflight "${args[@]}"
exec python3 -B -u runner/run_lane.py --phase run "${args[@]}"
