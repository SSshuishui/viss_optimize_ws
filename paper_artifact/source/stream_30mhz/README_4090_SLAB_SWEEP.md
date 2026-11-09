# 30 MHz stream-only slab-size sensitivity on 4×RTX 4090

This package is the 30 MHz `Nside=16384` stream-only implementation configured for a controlled slab-size sweep on four RTX 4090 GPUs.

## What is fixed

- Frequency: 30 MHz
- HEALPix: `Nside=16384`
- GPUs: `0,1,2,3`
- Day: day 1 by default
- Temporal segments: 10
- Reconstruction: representative-group mode (`occ_mode=2`)
- Visibility tile size: 256 pixels
- Orbit seed: 42

## Slab sizes

The default sweep uses:

- `33,554,432` pixels (`2^25`)
- `67,108,864` pixels (`2^26`)
- `134,217,728` pixels (`2^27`, manuscript setting)
- `268,435,456` pixels (`2^28`)

Only one day is run for every slab size. Use the same day for all configurations so the baseline geometry and occupied groups are controlled.

## Build

```bash
chmod +x build_4090.sh scripts/*.sh tools/*.py
./build_4090.sh
```

## Collect machine information

```bash
./scripts/collect_system_info.sh system_info_4090
```

## Smoke test: one slab size

```bash
export SKY_DIR=/path/to/earth_30Mhz
SKY_DIR="$SKY_DIR" \
GPUS=0,1,2,3 \
DAY=1 \
SLAB_PIX=134217728 \
WRITE_OUTPUT=0 \
./scripts/run_30m_one_day_4090.sh
```

## Full slab sweep

```bash
export SKY_DIR=/path/to/earth_30Mhz
nohup env \
SKY_DIR="$SKY_DIR" \
GPUS=0,1,2,3 \
DAY=1 \
SEGS=10 \
SLAB_LIST="33554432 67108864 134217728 268435456" \
REPEATS=1 \
WRITE_OUTPUT=0 \
OUT_ROOT=./results_30m_4090_slab_sweep \
./scripts/run_slab_sweep_4090.sh \
> slab_sweep_4090.log 2>&1 &
```

Monitor:

```bash
tail -f slab_sweep_4090.log
```

## Outputs

- `slab_sweep_summary.csv`: raw one-row-per-run measurements
- `slab_sensitivity_table.csv`: grouped summary
- `slab_sensitivity_table.md`: paper-ready table
- `day1_slab*/run.log`: complete logs
- `day1_slab*/gpu_info.txt`: GPU inventory seen by each run

The program records visibility time, reconstruction time, segment-loop time, total day time, the number of slabs, occupied groups, and observed peak GPU memory.

## Output-map policy

The sweep defaults to `WRITE_OUTPUT=0`, because each 30 MHz float32 map is about 12 GiB and the same final write would be repeated for every slab size. To preserve one output for a correctness check, run only the manuscript slab size once:

```bash
SKY_DIR="$SKY_DIR" GPUS=0,1,2,3 DAY=1 \
SLAB_PIX=134217728 WRITE_OUTPUT=1 \
OUT_ROOT=./results_30m_output_check \
./scripts/run_30m_one_day_4090.sh
```
