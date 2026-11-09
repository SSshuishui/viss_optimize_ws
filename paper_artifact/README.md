# Multi-GPU lunar-orbit radio-interferometric imaging

This repository provides the CUDA implementations, experiment drivers, numerical checks, and compact measurements for *Multi-GPU Acceleration for Data-Intensive Lunar-Orbit Radio Interferometric Imaging*. The `paper-artifact-v2` branch adds the major-revision experiments to the original `paper-artifact-v1` artifact. Cite the fixed release and full commit SHA recorded with the submitted revision. Citation metadata is in [CITATION.cff](CITATION.cff).

The original modules and measurements remain under `source`, `experiments`, `results`, and `figures`. Revision code is under `revision_20261109`. Its `agent_exchange` directory preserves the frozen implementations and relative dependencies of the evaluated experiments. The current server source snapshot has its own source identities. [MANIFEST.md](MANIFEST.md) maps each experiment to code and results, and [DATA.md](DATA.md) specifies the external inputs.

| Module | Entry point |
|---|---|
| Original exact-group and representative STS | `source/final_sts/build.sh` |
| Original controlled direct 3D | `source/direct_3d_reference/build.sh` |
| Original 30 MHz streaming | `source/stream_30mhz/build_4090.sh` |
| Five-day direct/STS inverse timing | `revision_20261109/build_artifact.py --component five_day` |
| Repeated segmentation/grid settings | `revision_20261109/build_artifact.py --component sweep` |
| Matched forward tile paths | `revision_20261109/build_artifact.py --component forward` |
| Corrected direct N/W and 450-day lanes | `revision_20261109/build_artifact.py --component direct450` |
| Mature direct and corrected-DCF export | `--component mature_direct` and `--component corrected_dcf` |
| CPU and masked Fourier controls | `--component cpu_nufft` |
| Fixed-microchunk streaming and MPI | `--component microchunk` and `--component mpi` |

On Linux, install CUDA 12.4, a CUDA-compatible OpenMP C++ compiler, Python 3.9 or later, and OpenSSL development headers/libraries. The reported 4090 builds used GCC 11.4 and `sm_89`. Direct 450-day portability also used `sm_75` and `sm_86`. MPI compilation additionally requires an installed `mpicxx` wrapper. Numerical Python analyses use NumPy. CPU NUFFT controls use FINUFFT 2.5.1, and GPU Fourier controls use CuPy and cuFINUFFT 2.5.1. Plotting requires Matplotlib and, for the older sensitivity plot, pandas.

Run the static package checks from the repository root. They compile Python syntax, run `bash -n`, resolve quoted headers, and verify published source fingerprints. They import no numerical drivers.

```bash
python3 -B validate_artifact.py
python3 -B revision_20261109/build_artifact.py --dry-run
```

Compile all revision components with an explicit compiler and GPU architecture. This command builds programs and CPU libraries without launching GPU experiments or reading sky arrays. It writes a build receipt under `revision_20261109/build`.

```bash
python3 -B revision_20261109/build_artifact.py \
  --arch sm_89 --nvcc /usr/local/cuda-12.4/bin/nvcc \
  --host-cxx /usr/bin/g++-11
```

Repeat `--component` to build a subset. For example, the following prepares the single-card 450-day code for an RTX 2080 Ti.

```bash
python3 -B revision_20261109/build_artifact.py \
  --component direct450 --arch sm_75 \
  --nvcc /usr/local/cuda-12.4/bin/nvcc --host-cxx /usr/bin/g++-11
```

[The revision reproduction guide](revision_20261109/README.md) gives cache generation, timing, CPU/Fourier, lane, and merge entry points. Historical benchmark drivers retain their numerical acceptance checks and require the corresponding external reference datasets. The public build entry compiles their frozen sources independently of the original server's scheduling markers.

The original strong-scaling and grouped-STS comparison can be run with the supplied sky input.

```bash
SKY_DIR=/path/to/earth_10Mhz ./experiments/run_scaling_10m.sh
SKY_DIR=/path/to/earth_10Mhz ./experiments/run_five_day_paired_10m.sh
```

The five-day script writes full-resolution maps for its accuracy comparison. Inspect its output and resource settings before running it. Absolute timings depend on the GPU, compiler, driver, NUMA placement, clocks, and shared-node conditions. Each recorded batch retains its own timer and output definition. Direct DCF numerators and cell-normalized STS responses have different estimators, and their timing ratios require the associated numerical comparisons.

The artifact contains source code, input identities, compact measured records, and plotting inputs. Sky maps, visibility caches, and full-resolution output arrays are external inputs or generated products. Their required formats, hashes, and scopes are documented in [DATA.md](DATA.md).
