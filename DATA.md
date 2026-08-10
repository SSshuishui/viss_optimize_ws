# Data notes

## Required sky inputs

- 10 MHz experiments: HEALPix `Nside=4096`, file `B_10M.bin`.
- 30 MHz experiments: HEALPix `Nside=16384`, file `B_30M.bin`.

The binary loader expects contiguous single-precision values in the format used
by the supplied eGSM preprocessing workflow. Before a public release, add the
upstream map version, generation command, and SHA-256 checksums here. Do not
redistribute an upstream map unless its terms permit it.

The direct 3D reference uses the compact DCF file
`source/direct_3d_reference/dcf_mb_10M_days450_seed42.bin`, which is included.

## Deliberately excluded files

- compiled CUDA executables;
- input sky maps;
- full-resolution reconstruction maps;
- Nsight `.nsys-rep` and `.sqlite` databases;
- notebook checkpoints and macOS metadata.

These exclusions keep the repository reviewable while retaining source,
configuration, compact logs, aggregate measurements, and figure inputs.

