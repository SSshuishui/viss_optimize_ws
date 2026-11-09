# R2.7 bounded forward controls

Prepared locally for a coordinator-assigned idle RTX 4090. Local preparation has not compiled CUDA or run a GPU kernel.

## Visibility tile ablation

Use the original 10 MHz day-1 eGSM/HEALPix NEST FP32 sky, all 201,326,592 directions at Nside=4096. Use the first 1,280 timestamps of the first 6,981-timestamp retained observing window, orbit seed 42, so both branches evaluate the same 35,840 independent half baselines. Both branches use one CUDA thread per half baseline, 256 threads per block, 140 blocks, 256 pixels per tile, and five shared FP32 arrays. Sky weights, endpoint normalization, lunar threshold, FP32 phase/trigonometry, sample order, arithmetic and output storage come from the frozen production headers.

A is the unmodified frozen half-set kernel with tile classification and its visible, hidden, and mixed paths. B is cloned from A with tile classification disabled. B runs the exact original mixed-path per-pixel computational mask for every pixel. The generated diff shows the two changed lines, including the distinct kernel name. The old T04 generic forward implementation is excluded because it uses a different phase expression and shared array layout.

Timing uses CUDA events around one launch. Common orbit generation, sky load, direction and tile metadata preparation, output transfers and comparison are outside the timed intervals. Tile metadata is prepared once for both branches, so the result measures the shortcut's kernel cost separately from building the metadata. One warm-up launch per branch precedes four recorded pairs in AB, BA, BA, AB order. The summary retains both the ratio of medians and the median of the paired ratios. Host/GPU clocks are sampled at the run boundaries without locking clocks.

Compare the complete raw complex visibility vectors before phase correction, using relative L2, maximum absolute difference, peak-normalized maximum difference, and count of changed complex FP32 words. The predeclared closeness threshold is 1e-5 for both relative L2 and peak-normalized maximum difference. All formal repeats must also be bitwise identical within each branch. A failed gate retains observations and returns a distinct status. This comparison checks the computational mask and both branch outputs for the selected real window. It does not measure a separate speedup for hidden versus visible classification or compare with a higher-precision physical visibility reference.

## Geometry preparation diagnostic

Start from the same day-1 orbit state precomputed with the original generator, and use the identical 1,280-timestamp window. The online branch launches the frozen position-gather, signed UVW and endpoint-generation, and endpoint-inverse-norm kernels to prepare device-ready arrays. The replay branch reads the same nine generated signed FP32 arrays from a prerecorded binary through the warmed filesystem cache, transfers them into the same device arrays, and computes the same endpoint inverse norms. Compare all 11 output arrays byte-for-byte to the frozen generated geometry.

There is one untimed warmup of each branch, followed by eight pairs alternating order. The serial wall timer runs from branch dispatch through device-ready synchronization and includes file open/read in the replay branch. One-time orbit precompute, snapshot creation, buffer allocation, validation, D2H transfers for verification and all visibility computation are outside this comparison. This is a matched device-ready preparation diagnostic. Production host-copy, multi-GPU distribution, overlapped scheduling, historical text parsing and cold filesystem reads are outside the protocol. Record original runtime cost separately from the resulting file-versus-generation timing.

## Deployment by coordinator

Run from a new server copy of this directory on the authorized host.

```bash
bash build.sh
bash run.sh ASSIGNED_GPU_UUID /data/zhaox/research/viss_optimize_ws/earth_10Mhz/B_10M.bin NEW_ABSOLUTE_RUN_DIRECTORY
```

Use the assigned GPU UUID and a previously absent output directory. The input file is read in place and checked against the previously frozen SHA-256. The launcher checks active compute processes on the selected device and caps the binary at 150 seconds plus a 10-second termination grace period. No retry is automatic. The retained files consist of small raw visibility arrays, a 2.58 MB geometry snapshot, timing and validation CSVs, source/build/input hashes, and logs. The sky and any image products are excluded.

Frozen headers were copied from `parco_revision_agent_exchange/07_performance_system/T04_20261101_smallgate/frozen`. The model sky identity is the SHA-256 in the T19 A100/4090 portability evidence. CUDA 12.4, `-O3`, `--use_fast_math`, and `sm_89` mirror the T04 build. The bounds and observations apply to this workload and single-GPU window.
