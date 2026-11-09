#!/usr/bin/env bash
set -euo pipefail
batch_root="$(cd "$(dirname "$0")/.." && pwd)"
nvcc_path="${NVCC:-/usr/local/cuda-12.4/bin/nvcc}"
python3 - "$batch_root" <<'PY'
from pathlib import Path
import json,sys
root=Path(sys.argv[1]);formal=root.parent.parent/'07_performance_system/T07_20261101_legacy_inverse_comparison/status.json'
state=json.loads(formal.read_text())
if state.get('state')!='completed' or state.get('formal')!=16:
    raise SystemExit('formal batch has not completed/released; compilation is intentionally deferred')
if not (root/'FULLREF_ALLOWED').is_file():
    raise SystemExit('root-agent resource clearance/authorization marker FULLREF_ALLOWED required')
PY
mkdir -p "$batch_root/bin"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DLEGACY_FASTPATH=1 "$batch_root/scripts/replay_corrected_dcf_full.cu" -lcrypto -o "$batch_root/bin/replay_corrected_dcf_full"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp "$batch_root/scripts/replay_sts_export.cu" -lcrypto -o "$batch_root/bin/replay_sts_export"
"$nvcc_path" -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_89 -Xcompiler -fopenmp -DOP_RECON_SKIP_ZERO_WEIGHTED_VISS=1 -DOP_RECON_DEBUG_CHECK_FINITE=1 "$batch_root/scripts/zero_v_control.cu" -o "$batch_root/bin/zero_v_control"
