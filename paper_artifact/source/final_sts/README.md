# Final STS method: sensitivity and profiling version

This directory contains only the final `mirror_tile_vissfast` method, with experimental controls added for the requested revision.

## What was changed

1. **Temporal-segment sensitivity**
   - Existing `--segs` is retained.
   - Added `--time_start` and `--time_count` so a controlled short observation window can be run identically in STS and global-3D code.
   - Each segment now reports the local-plane fitting diagnostics
     - `fit_rmse_wavelength`: RMS of `w - fa*u - fb*v`;
     - `fit_max_abs_wavelength`: maximum absolute residual.

2. **Grid-cell sensitivity**
   - Added `--grid_cell_scale`.
   - `1.0` reproduces the original grid construction.
   - `<1.0` creates finer `(u,v)` cells: more occupied groups, usually slower, but less grid quantization/group spread.
   - `>1.0` creates coarser cells: fewer groups, usually faster, but a larger approximation risk.
   - The code keeps the same covered `(u,v)` range and adjusts `RES` to an odd value. Both requested and actual scale are written to the profile CSV.

3. **Profiling**
   - Added per-segment CUDA-event timing for:
     - visibility kernel;
     - visibility D2H;
     - reduced visibility H2D;
     - reconstruction kernel.
   - Added host timing for:
     - host visibility reduction and phase correction;
     - group construction;
     - complete visibility/reconstruction stages;
     - final output D2H and file writing.
   - Added observed peak GPU memory from `cudaMemGetInfo`.
   - Added optional NVTX ranges for Nsight Systems.

4. **Output and experiment control**
   - `--profile=0/1`
   - `--write_output=0/1`
   - `--run_tag=<tag>` to avoid overwriting results.
   - Exact group-wise STS uses `--occ_mode=0`; representative median-baseline STS uses `--occ_mode=2`.

5. **Correctness/engineering fixes**
   - The sky loader now uses `B_<btag>.bin/txt`, rather than hard-coding `B_10M`.
   - Temporary exact-group `d_offsets` and `d_idx` allocations are freed after every segment.

## Why vary segment count and grid-cell size?

The local plane is fitted independently in every temporal segment:

`w ≈ fa*u + fb*v`.

Increasing `--segs` makes each segment shorter. This normally lowers the plane-fit residual and makes member baselines inside a group more similar, but it increases scheduling and metadata overhead.

`--grid_cell_scale` changes the local `(u,v)` quantization width. A coarser cell merges more baselines into one reconstruction group and reduces work. A finer cell reduces grouping/representative error but creates more groups and more reconstruction work.

Use two one-dimensional sweeps rather than a full Cartesian sweep:

- segment sweep: `segs = 5, 10, 20, 40`, fixed grid scale `1.0`;
- grid sweep: `scale = 0.75, 1.0, 1.25, 1.5`, fixed `segs = 10`.

## Build

```bash
./build.sh
```

Default target is `sm_89`. Override with `ARCH=sm_80`, etc.

For Nsight/NVTX:

```bash
ENABLE_NVTX=1 ./build.sh
```

## Run one configuration

```bash
SKY_DIR=/path/to/earth_10Mhz \
GPUS=0,1,2,3 \
SEGS=10 GRID_CELL_SCALE=1.0 OCC_MODE=2 \
OUT_DIR=./results/example RUN_TAG=example \
./scripts/run_one.sh
```

For a short controlled workload, add, for example:

```bash
TIME_START=0 TIME_COUNT=256 NSIDE=256 SKY_DIR=/path/to/nside256_sky ./scripts/run_one.sh
```

The sky file must match the selected `NSIDE`.

## Sensitivity sweeps

```bash
SKY_DIR=/path/to/earth_10Mhz GPUS=0,1,2,3 ./scripts/run_segment_sweep.sh
SKY_DIR=/path/to/earth_10Mhz GPUS=0,1,2,3 ./scripts/run_grid_sweep.sh
```

By default each configuration runs both exact group-wise STS and representative STS, then computes Pearson correlation, raw NRMSE, affine-aligned NRMSE, and relative L2 error. Set `RUN_EXACT=0` for timing-only sweeps.

## Profiling

Program-level profiling CSV is enabled by default. For a 10 MHz Nsight run:

```bash
SKY_DIR=/path/to/earth_10Mhz GPUS=0,1,2,3 ./scripts/profile_10mhz_nsys.sh
```

The Nsight trace should be used for the paper's overlap percentage. The CSV provides component timings, but it deliberately does not infer an overlap ratio from summed timers because that would mix inter-GPU concurrency with actual transfer/compute overlap.

## Map comparison

```bash
python3 tools/compare_maps.py \
  --reference exact.bin \
  --candidate representative.bin \
  --json-out metrics.json \
  --csv-out metrics.csv
```

The script uses `numpy.memmap`, so it does not load both full maps into RAM.
