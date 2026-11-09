# Controlled 10 MHz 3D+DCF baseline

This package is based on `system_viss_recon_from_3d_as_baseline.zip`.

## Fixed controlled configuration

- 10 MHz, HEALPix Nside=4096
- day 1, complete 69,810 time samples
- `segs=10`
- one RTX 4090 by default (`GPUS=0`)
- sky input: `B_10M.bin` in the directory supplied through `SKY_DIR`
- online orbit/baseline generation, `orbit_seed=42`
- half-baseline visibility evaluation and phase correction
- exact endpoint occultation in 3D reconstruction
- DCF file: `dcf_mb_10M_days450_seed42.bin`
- visibility tile size 256, aligned with the profiled final-method configuration

## Build and run

```bash
chmod +x build.sh scripts/*.sh tools/*.py
./build.sh
SKY_DIR=/path/to/earth_10Mhz \
  nohup ./scripts/run_3d_dcf_10m_1gpu.sh > 3d_dcf_10m_1gpu.log 2>&1 &
tail -f 3d_dcf_10m_1gpu.log
```

Output:

```text
out_3d_dcf_10m_1gpu/C1day10M_3d_dcf_1gpu.bin
```

## Exact-group STS on the same GPU

```bash
SKY_DIR=/path/to/earth_10Mhz \
  nohup ./scripts/run_exact_sts_10m_1gpu.sh > exact_sts_10m_1gpu.log 2>&1 &
```

The helper uses the adjacent `source/final_sts` directory by default. Override
it with `FINAL_METHOD_DIR=/other/path` when needed.

For a fair one-GPU timing table, also rerun representative-group STS:

```bash
SKY_DIR=/path/to/earth_10Mhz \
  nohup ./scripts/run_rep_sts_10m_1gpu.sh > rep_sts_10m_1gpu.log 2>&1 &
```

## Important comparison boundary

The forward operator, sky input, online orbital parameters, seed, day, and segment split are aligned. The 3D package uses a fused GPU visibility reduction/phase-correction path, while the sensitivity package may report the same mathematical forward result through a different reduction implementation. Use reconstruction-stage time as the primary controlled method comparison. Whole-pipeline time can be reported only with this implementation detail stated.
