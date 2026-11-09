# Run on the coordinated server

The new directory can be copied independently. Reused sources and data are
read from their frozen archive paths. The runner writes only under this
directory. Use the coordinated idle GPU UUID.

```bash
EXCHANGE=/data/zhaox/research/parco_revision_20261026/agent_exchange
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
# Set GPU_UUID to the coordinated idle RTX 4090's complete UUID.
bash build.sh "$EXCHANGE"
python3 -u run_five_day_timing.py --exchange-root "$EXCHANGE" --gpu-uuid "$GPU_UUID" --phase preflight
python3 -u run_five_day_timing.py --exchange-root "$EXCHANGE" --gpu-uuid "$GPU_UUID" --phase run
```

Use the same arguments after a clean stop at a run boundary. `run/PAUSE`
requests a stop at the next boundary. A failed in-progress attempt remains
under `run/attempts/` and requires inspection before retrying. To override
archived locations, use `--cache-root` or `--dcf-bin` with the original
SHA-identical files. `--run-dir` accepts a separate direct child of this
directory. Python 3.10+ and NumPy are needed for formal numerical checks.
The build requires CUDA 12.4, an sm_89 target and OpenSSL development
headers. Preserve the build log. Preflight verifies the build and the
complete source/cache/reference identities before run.

Use `run/status.json` for progress, `run/formal_runs.csv` for ordered raw
times, `run/summary.json` for per-day medians and paired cost ratios,
and `run/cases/*/segments.csv` for stage timing. Existing day-1 stage
measurements live in the old batch's `metrics/stage_sums.csv`. The
same-binary 2026-11-01 reference values were B 174.248281 s and C
77.237995 s. The follow-up uses rebuilt date-parameterized wrappers.
