# MPI/CUDA STS implementation

The driver assigns one CUDA GPU per MPI rank and disjoint complete 256-pixel tiles to each rank. Rank zero generates geometry, reduces visibility data and constructs groups. Flat and hierarchical strategies use host-buffer MPI. The supplied vendor kernels and single-node reference retain their recorded source identities.

Build with a consistent CUDA, C++17 and MPI toolchain.

```sh
CUDA_ARCHS='86;89' NVCC=/path/to/nvcc MPICXX=/path/to/mpicxx bash scripts/build.sh
```

The build supports Open MPI and MPICH wrapper queries. `--skip-reference` omits the separate single-node reference build. The optional `bootstrap_mpich.py` installs a user-supplied MPICH wheel into an isolated prefix. External MPI/CUDA dependencies are resolved on the executing platform.

Start with a synthetic small problem and an explicit available GPU. The launcher exposes its complete settings through `--help`.

```sh
python3 scripts/launch_background.py --help
python3 scripts/launch_background.py --mode local-smoke --physical-gpu=0 --ranks=1,2 --recon-modes=rep,exact --run-dir runs/new_local_check
```

The local smoke test can share one device for correctness checks. Use actual separate devices and nodes for scaling measurements. The `config/*.example` and `config/*example.json` files contain placeholders. Configure hostfiles and node records for the intended site before cluster execution. `--prepare-only` writes a command plan without executing MPI or CUDA.

`tests/compare_runs.py` compares numerical outputs. `tests/run_smoke_matrix.py` also compares with the included production reference. The host-only test fixtures and MPI transport probe are included. `scripts/package.py` and `scripts/verify_package.py` generate and check independent source-package manifests.

The distribution contains project source and placeholder configuration. Site-specific rental deployment, relay tools, credentials, run arrays and prebuilt executables are excluded. `PUBLIC_SOURCE_FILES.json` maps copied files to their original workspace-relative paths and SHA-256.
