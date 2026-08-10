# Lunar-orbit radio-interferometric imaging: paper artifact

This repository is the curated code-and-results artifact for the manuscript
"A Data-Intensive Multi-GPU Pipeline for Lunar-Orbit Radio Interferometric
Imaging with Approximation-Controlled Short-Time Stacking."

It contains the CUDA/C++ implementations used in the controlled experiments,
the launch and analysis scripts, compact raw logs and CSV/JSON summaries, and
the script and data used to generate the sensitivity figure. Compiled
executables, profiler databases, input sky maps, and full-resolution output
maps are intentionally excluded.

## Layout

- `source/final_sts`: exact-group and representative-group STS implementation.
- `source/direct_3d_reference`: direct 3D+DCF reference implementation and DCF.
- `source/stream_30mhz`: memory-bounded 30 MHz streaming implementation.
- `experiments`: portable wrappers for the five-day comparison and 1/2/4-GPU scaling.
- `results`: compact records from the controlled RTX 4090 experiments.
- `results/summary`: normalized paper-facing averages and scaling values.
- `figures`: Figure 4 source script and generated PNG.

See `MANIFEST.md` for the mapping between manuscript experiments and files.

## Requirements

The reported runs used Linux, CUDA 12.4, GCC 11.4, and four RTX 4090 GPUs. The
code defaults to CUDA architecture `sm_89`; override `ARCH` when building for a
different supported GPU.

Python analysis scripts require Python 3 and NumPy. Figure generation also
requires Matplotlib.

## Input data

Set `SKY_DIR` to a directory containing `B_10M.bin` or `B_30M.bin`, as required
by the run. The maps are not included because of redistribution terms and file
size. See `DATA.md` for input identifiers and output policy.

## Quick start

Build the final STS implementation:

```bash
cd source/final_sts
./build.sh
```

Run the 10 MHz default configuration:

```bash
SKY_DIR=/path/to/earth_10Mhz GPUS=0,1,2,3 \
  WRITE_OUTPUT=0 ./scripts/run_one.sh
```

Run the compact experiment wrappers from the repository root:

```bash
SKY_DIR=/path/to/earth_10Mhz ./experiments/run_scaling_10m.sh
SKY_DIR=/path/to/earth_10Mhz ./experiments/run_five_day_paired_10m.sh
```

The paired script writes two full-resolution maps per day by default so that it
can compute reconstruction metrics. Set `WRITE_OUTPUT=0` only for timing-only
runs.

Regenerate Figure 4:

```bash
python3 figures/plot_sensitivity_tradeoff.py
```

## Scope and provenance

The artifact preserves measured outputs rather than claiming that all results
can be reproduced on a non-identical machine. Absolute timings depend on GPU,
driver, compiler, filesystem, and NUMA placement. Historical logs retain the
paths recorded on the evaluation server; runnable scripts in this repository
use relative paths or explicit environment variables.

No license is asserted here. A license should be added only after the authors
confirm that the source and derived inputs may be redistributed under it.

