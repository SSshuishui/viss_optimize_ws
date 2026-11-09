# Production microchunk regression source

This component contains the complete source and source-generation script for the 30 MHz full-field first-segment regression. The inverse kernels are retained. MODE4 uses fixed 32,768-pixel FP32 arithmetic microchunks and persistent FP64 outer accumulation, with one final cast per device partition.

Build on a CUDA-capable Linux host with C++17, OpenMP and OpenSSL development files.

```sh
NVCC=/path/to/nvcc bash scripts/build.sh
```

The build runs `prepare_source.py` against the included frozen `source/original_main.txt`. All quoted CUDA/C++ includes are included. The sibling short30 and consumed-representative source files required by `cpu_source_qa.py` are included under `agent_exchange/09_memory_boundary/`.

`PROTOCOL.json` identifies the scientific input, dimensions, numerical gates and original device assignments. `scripts/run_regression.py` is the archived experiment controller. It binds the recorded GPU UUIDs, full sky input, reference hashes, resource ledger and execution markers. A new execution requires an explicit site configuration and compatible reference data. Its source is supplied for reproducibility and inspection.

`REFERENCES_REQUIRED.json` identifies the separately supplied generated reference arrays. Place matching arrays under `references/` to use the archived numerical validation. The input sky is also supplied separately. The general numerical kernels and source build do not require these reference arrays. The historical resource-handoff branch additionally refers to its original external queue checker and completed-run ledgers. Those execution-state artifacts are outside this source distribution.

`scripts/cpu_inverse_oracle.py` and `scripts/finalize_regression.py` analyze completed cases. Host-only checkpoint and orchestration fixtures are included. `PUBLIC_SOURCE_FILES.json` maps each copied source to its original workspace-relative path and SHA-256.
