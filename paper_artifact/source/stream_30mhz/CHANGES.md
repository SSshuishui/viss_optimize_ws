# Changes from the supplied stream-only version

- Added an RTX 4090 build script using `-arch=sm_89`.
- Replaced the old example configuration (`day_start=406`, `day_count=10`, two GPUs) with controlled one-day, four-GPU launch scripts.
- Added a sequential slab-size sweep for `2^25`, `2^26`, `2^27`, and `2^28` pixels.
- Added `--write_output=0/1` so timing sweeps do not write a roughly 12 GiB map for every configuration.
- Added `--run_tag` to avoid output-name collisions.
- Added `--summary_csv` and a machine-readable per-run summary.
- Added observed peak GPU-memory sampling with `cudaMemGetInfo`.
- Added total slab counts, visibility/reconstruction totals, segment-loop time, output time, and total day time.
- Added automatic CSV and Markdown table generation.
- Added system-information collection for CPU, RAM, storage, CUDA, driver, GPU inventory, and GPU topology.
- Fixed release of temporary exact-group reconstruction arrays after each segment. This does not alter representative-group (`occ_mode=2`) results.
