# 30 MHz RTX 4090 Slab-Size Sensitivity Analysis

## Experimental scope

- Frequency: 30 MHz
- HEALPix: Nside=16384
- Day: 1
- Temporal segments: 10
- GPUs: 4 × NVIDIA GeForce RTX 4090
- Output writing: disabled
- Repeats per slab size: 1

## Results

| Slab pixels | Total slabs | Visibility (s) | Reconstruction (s) | Total compute day (s) | Relative runtime | Peak GPU memory (GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 33,554,432 | 96 | 1844.94 | 1053.08 | 2898.29 | 1.0000 | 7.65 |
| 67,108,864 | 48 | 1846.44 | 1053.02 | 2899.72 | 1.0005 | 8.53 |
| 134,217,728 | 24 | 1852.62 | 1052.93 | 2905.80 | 1.0026 | 10.28 |
| 268,435,456 | 12 | 1854.17 | 1052.58 | 2907.02 | 1.0030 | 13.79 |

## Main findings

1. Total runtime is insensitive to slab size over the tested range. The spread is 8.73 s, or 0.30% of the fastest run.
2. Reconstruction time changes by only 0.50 s (less than 0.05%). All runs have the same total occupied-group count, 1,293,346.
3. Peak GPU memory grows approximately linearly with slab size. A linear fit gives about 6.77 GiB fixed memory plus 28.07 bytes per slab pixel.
4. The 134,217,728-pixel setting is a practical balance: 24 total slabs, 10.28 GiB peak memory, and only a 0.26% runtime increase relative to the smallest slab.
5. The largest slab does not improve runtime but adds about 3.51 GiB peak memory relative to 134,217,728 pixels.
6. Because each point was run once and the sweep order was ascending, do not claim that the smallest slab is statistically faster. State that runtime is stable within 0.3%.

## Suggested manuscript paragraph

Across slab sizes from 33.6 million to 268.4 million pixels, the 30 MHz compute time remained nearly unchanged, varying from 2898.29 s to 2907.02 s (a 0.30% range), while peak GPU memory increased from 7.65 GiB to 13.79 GiB. Reconstruction time varied by less than 0.05% because all configurations processed the same 1,293,346 occupied groups. We therefore use a slab size of 134,217,728 pixels as a balanced configuration: it reduces the full-sky pass to 24 slabs, requires 10.28 GiB peak memory per GPU, and incurs only a 0.26% runtime difference relative to the smallest tested slab.

## Hardware summary

- CPU: 2 × Intel Xeon Gold 6530, 16 cores/socket, 32 physical cores, 64 hardware threads
- RAM: 248 GiB
- GPU: 4 × RTX 4090, 24,564 MiB each
- Driver: 560.35.03
- CUDA compiler: 12.4 (V12.4.99)
- GCC: 11.4.0
- Linux kernel: 5.15.0-141-generic
- GPU topology: no NVLink; GPUs 0–2 are attached to NUMA node 0 and GPU 3 to NUMA node 1; connections are reported as NODE/SYS
- PCIe capability: Gen4 ×16 maximum; the collected idle snapshot reported Gen1 current state, which should not be used as the active-workload link rate

## Remaining measurement needed

The experiment path's actual storage device is not established by the collected system files. Record it with:

```bash
findmnt -T .
df -Th .
```

For a final end-to-end 30 MHz table that includes output I/O, run the selected 134,217,728-pixel configuration once with output writing enabled.
