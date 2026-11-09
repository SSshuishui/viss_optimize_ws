# Corrected direct reference lanes

Use sibling lane directories under the selected data directory and one physical GPU UUID per lane. Assign explicit, disjoint dates across lanes. The same list, UUID, executable paths, and frozen inputs are required when resuming. Linux, `nvidia-smi`, Python 3.9 or later, and NumPy are required. The forward executable is the UUID-enabled frozen production cache exporter. The inverse executable is `direct_daily` from the companion source directory.

```bash
python3 /absolute/path/to/run_lane.py \
  --phase preflight \
  --lane-root /data/project/direct450/lane-01 \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 \
  --dates 1-449:2 \
  --forward-bin /data/project/direct450/bin/main_cache \
  --direct-bin /data/project/direct450/bin/direct_daily \
  --sky-file /data/project/sky/B_10M.bin \
  --dcf-bin /data/project/dcf/dcf_mb_10M_days450_seed42.bin
```

After the portability gate, run the same command with `--phase run`. A new lane directory must be empty. Redirect the console log to a sibling directory or preflight before opening a console log in the lane. Optional `--build-receipt` records the build receipt path and SHA-256 with the executable identities. An empty `PAUSE` file stops the lane at the next day boundary. Remove it and rerun the same command to resume.

Each date regenerates ten one-GPU `PREPLAY1` segments at 10 MHz and Nside 4096 with orbit seed 42, blockage enabled, occultation mode 2, visibility tile 256, and grid scale 1. The runner checks 1,954,680 half-set records per day, per-segment geometry and visibility fingerprints, the inverse's cache and DCF identities, and the raw numerator and denominator sizes, SHA-256 and finiteness. Each stage retains its wall time, command and diagnostics. The inverse compute-loop time is reported separately from process wall time and disk checkpoint time.

Only `CURRENT.json` selects the durable lane generation. Its manifest contains the exact accepted date prefix and two flat little-endian FP64 arrays, `N.f64` and `W.f64`, in NEST order. Dates enter the pool only after both new arrays are written and synced. Once the new pointer is synced, the runner removes the previous generation and that date's caches and raw maps. Failed or interrupted attempts retain compact diagnostics in `receipts/`. The runner can rebuild an interrupted date from the last verified generation. It never adds per-day normalized maps.

The default disk guard requires 12 GiB available per invocation. Concurrent lanes on one filesystem need enough space for their combined daily scratch, transient replacement generations, and checkpoints. Each final checkpoint uses 3 GiB, and each lane can reach about 8.4 GiB while replacing a checkpoint. The runner checks for at least 8500 MiB of free device memory before each GPU process, checks for foreign compute contexts before each GPU process and every ten seconds during execution, and shares a per-UUID lock among sibling lanes via `.gpu_locks` in their parent. A conflicting third-party context stops the current stage. Only the runner's own subprocess group is terminated.
