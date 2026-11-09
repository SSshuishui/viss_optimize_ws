# Repeating three original STS sweep settings

Copy this directory into its own follow-up server path. All archive inputs
are read-only and new outputs are confined to this directory. Use an idle
RTX 4090 coordinated for this experiment and available throughout the batch.

```bash
EXCHANGE=/data/zhaox/research/parco_revision_20261026/agent_exchange
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
# Set GPU_UUID to the full UUID of the assigned RTX 4090.
bash build.sh "$EXCHANGE"
python3 -u run_sweep_repeats.py --exchange-root "$EXCHANGE" --gpu-uuid "$GPU_UUID" --phase preflight
python3 -u run_sweep_repeats.py --exchange-root "$EXCHANGE" --gpu-uuid "$GPU_UUID" --phase run
```

Use `--cache-root` with a different archive location only if all ten cache
files have their frozen SHA256. Runtime needs NumPy and CUDA. The build
requires the CUDA 12.4 compiler and OpenSSL development headers.
The native CUDA entry point accepts only the three protocol settings.

Outputs include `run/freeze.json` (the input and resegmentation identity),
`run/formal_runs.csv` (ordered individual timings), `run/summary.json`
(per-case descriptive statistics), and compact per-case logs, samples,
metadata and hashes. To request a clean stop at the next boundary, set
`run/PAUSE`. Resume with the same arguments after clearing that marker
and resolving any reported collision. A partial set of formal observations
receives separately labeled resume warm-ups. An unfinished attempt is
retained and must be reviewed before retrying.
