# Daily corrected DCF direct 3D inverse

The runner reuses the archived T01 weighted inverse headers. Verify their exact
contents with `FROZEN_SHA256SUMS`. It accepts the original 10 MHz, 4096 Nside
G1 PREPLAY1 cache format with ten consecutive 6981-timestamp segments and 28
independent pairs per timestamp. The table hash, 450-day table scope, band and
orbit seed are checked before execution.

For each segment the runner normalizes the cached endpoint vectors, computes
the frozen DCF pair weight, and multiplies the cached consumed-n visibility by
that weight once. The archived plan and mixed-tile kernels apply the same
visibility predicate and full three-dimensional phase. The kernel accumulates
the real numerator in FP32 and the weighted support in FP64. The host exports
both accumulators directly. Divide their separately summed long-horizon values
once to produce the final normalized dirty image. Pixels with zero total
support need a separate support flag or a declared zero-support convention.

## Build and run

Choose `sm_75` for RTX 2080 Ti, `sm_86` for RTX A6000, or `sm_89` for the
archived architecture. The build uses C++17, OpenMP, CUDA CUB and OpenSSL
libcrypto. Additional NVCC flags can select the site compiler and OpenSSL
paths.

```sh
CUDA_ARCH=sm_75 NVCC=/path/to/nvcc ./build.sh /path/to/bin/direct_daily \
  -ccbin /path/to/g++
mkdir /path/to/new_day_attempt
/path/to/bin/direct_daily --gpus=GPU-UUID --nside=4096 --day=1 --max_segments=10 \
  --dcf_bin=/path/to/dcf_mb_10M_days450_seed42.bin \
  --cache_root=/path/to/g1_cache --out_dir=/path/to/new_day_attempt
```

The output directory must already exist and each attempt must have its own
directory. The GPU argument may be omitted when CUDA already sees exactly one
device. The program estimates the full-field device allocation and checks
free device memory before initializing it.

## Outputs

- `raw_N.f32` and `W.f64` contain 201,326,592 little-endian NEST-ordered
  pixels each. Their exact sizes are 805,306,368 and 1,610,612,736 bytes.
- `full_stats.json` records byte and finite counts, extrema, zero counts and
  SHA-256 for each raw array. The runner rejects nonfinite values and negative
  weights. Zero weight is a valid no-support result.
- `segments.csv` contains the segment identity and source cache SHA-256,
  transfer and inverse timings, and visible and mixed task counts.
- `metrics.json` records cache, table, GPU and image identity and day-level
  timers. The inverse loop includes host-device copies, preparation and inverse
  execution. It excludes initial cache decode and raw-output export. Reported
  device use samples initialization and segment boundaries.

A successful attempt renames the files after closing them and publishes
`metrics.json` last. A collector can verify all hashes and then remove the
daily raw files after committing their sums. To check portability, compare the
archived day-1 raw N and W values at the 8192 frozen NEST pixel IDs on each GPU
architecture. FP32 atomics and architecture-dependent mask roundoff can affect
bitwise equality. The direct and forward portability gates should be recorded
with their separate input identities and tolerances.
