# Revision reproduction guide

Run commands from the repository root. The directory structure below preserves the original frozen experiment dependencies. `PUBLIC_SOURCE_FILES.json` maps copied files to their source identities. Component manifests cover MPI and microchunk sources. Three CPU/Fourier path adaptations are recorded with both original and published hashes. They replace author-specific input roots with explicit input paths and preserve the numerical operations and gates.

```bash
artifact_root=$(pwd)
exchange="$artifact_root/revision_20261109/agent_exchange"
followup="$exchange/followup_20261105"
```

Build entry points and static checks are described in the top-level README. `build_artifact.py` compiles frozen numerical implementations and writes generated executables to ignored `build` or `bin` directories. It records the selected architecture and compiler commands. Historical `build_reference.sh` and benchmark controllers preserve their original scheduling markers. Use the public build entry for a new installation.

The daily forward exporter can regenerate a production cache from the externally supplied 10 MHz input. Select an idle physical GPU by its full UUID and a new writable output directory. The frozen configuration uses orbit seed 42, ten temporal segments, half-set visibility, occultation mode 2, grid scale 1, and 256-pixel tiles.

```bash
mkdir -p /path/to/new-work/day1/cache
"$followup/direct450/bin/forward_uuid" \
  --gpus=GPU-00000000-0000-0000-0000-000000000000 \
  --btag=10M --nside=4096 --day_start=1 --day_count=1 --segs=10 \
  --orbit_seed=42 --gen_gpu_index=0 --time_start=0 --time_count=0 \
  --blockage=1 --occ_mode=2 --viss_tile_pix=256 --grid_cell_scale=1 \
  --B_mode=bin --C_mode=bin --sky_dir=/path/to/earth_10Mhz \
  --cache_root=/path/to/new-work/day1/cache --out_dir=/path/to/new-work/day1 \
  --cache_only=1 --write_output=0 --write_viss=0 --write_baseline_txt=0 --profile=0
```

The five-day timing driver consumes accepted production caches for days 1, 113, 225, 337, and 450. `deploy/five_day_timing/input_identities.json` lists all fifty cache identities and the external fixed-ID reference datasets. Supply them at the recorded exchange-relative paths and pass the cache root explicitly. Its direct output is a raw DCF numerator, while STS uses equal-cell normalization. The loop timings and finalization retain their separate scopes.

```bash
python3 "$followup/deploy/five_day_timing/run_five_day_timing.py" \
  --exchange-root "$exchange" --cache-root /path/to/accepted/cache \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 --phase preflight
# Repeat the command with --phase run after preflight succeeds.
```

The sweep driver uses the complete day-1 row payload for the three declared `(segments, scale)` settings `(10,1)`, `(20,1)`, and `(10,1.5)`. It repartitions records without changing their bytes or order. Its `input_identities.json` also identifies the external sample/group references. The sampled replay in `01_common_operator/T01_20261101_production_replay/source/replay.cu` supplies the corresponding group and sampled-output diagnostics. Creating a new reference set requires an independently checked configuration and an explicitly recorded new input manifest.

```bash
python3 "$followup/deploy/sweep_repeats/run_sweep_repeats.py" \
  --exchange-root "$exchange" --cache-root /path/to/accepted/day1-cache \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 --phase preflight
# Repeat the command with --phase run after preflight succeeds.
```

The matched forward control is self-contained once the exact sky input is supplied. It evaluates the first 1280 timestamps and all input directions. The public build creates the source/binary fingerprints consumed by its runner.

```bash
bash "$followup/deploy/forward_ablation/run.sh" \
  GPU-00000000-0000-0000-0000-000000000000 \
  /path/to/earth_10Mhz/B_10M.bin /path/to/new-forward-control
```

For a new corrected direct accumulation, use `direct450/runner/run_lane.py`. Each lane takes one full UUID, an explicit date list, the frozen sky and DCF, and its own initially empty directory. Run preflight first, then change `--phase preflight` to `--phase run`. Partition dates into disjoint lists when using several GPUs. The runner checks active contexts, verifies daily caches and raw outputs, commits FP64 N/W accumulators transactionally, and removes validated daily scratch.

```bash
python3 "$followup/direct450/runner/run_lane.py" \
  --phase preflight --lane-root /path/to/work/lane-01 \
  --gpu-uuid GPU-00000000-0000-0000-0000-000000000000 --dates 1-450 \
  --forward-bin "$followup/direct450/bin/forward_uuid" \
  --direct-bin "$followup/direct450/bin/direct_daily" \
  --sky-file /path/to/earth_10Mhz/B_10M.bin \
  --dcf-bin "$artifact_root/source/direct_3d_reference/dcf_mb_10M_days450_seed42.bin"
```

`direct450/merge/merge_direct450.py --help` lists the final merge arguments. Supply explicit completed lane manifests and the independently identified historical STS arrays when computing that comparison. The merge rejects incomplete lanes, duplicated dates, altered arrays, and mismatched scientific protocols. It pools N and W before one normalization. The historical STS and regenerated direct runs share the documented sky/orbit configuration. Historical per-segment visibility byte identities were unavailable.

The CPU/Fourier controls compile with `--component cpu_nufft`. For the matched CPU, transparent GPU-direct, and unmasked CPU FINUFFT control, provide the frozen geometry export and coordinate/brightness prefix directory to the explicit arguments.

```bash
python3 "$exchange/05_baselines/T05_20261101_matched/scripts/matched.py" \
  --geometry /path/to/geometry --prefix /path/to/prefix_dump \
  --output /path/to/new-cpu-control --cpu-only
```

For the exact-mask ROI control, the external data root contains the recorded `geometry`, `precision/queued_correctness_v2/pilot8_dump_v2/prefix_dump`, and `agent_exchange/01_common_operator/T01_20261101_production_replay/cache` inputs. The exporter validates geometry and selected cache rows. It writes the derived NPZ to standard output. The next command builds the 36 declared mask cases on CPU.

```bash
roi="$exchange/05_baselines/T22_20261101_masked_roi"
python3 "$roi/scripts/export_existing_inputs.py" /path/to/external-data-root > "$roi/inputs_existing.npz"
python3 "$roi/scripts/mask_statistics.py" --root "$roi" \
  --geometry /path/to/geometry/production_fp32_geometry_export_v1.npz
python3 "$roi/scripts/run_masked_roi.py" --root "$roi"
```

The last command prepares the CPU references. An explicit `--allow-gpu`, selected `--gpu` and matching `--expected-uuid` enable the archived GPU comparison. The original driver restricts GPU index choices to its declared evaluation window. Its one-warm-up/ten-apply protocol, finite-domain masks, tolerances and all-case retention remain in the source and `PROTOCOL.json`. The unmasked GPU ROI driver takes its external input root from `PARCO_REVISION_DATA_ROOT` and additionally consumes the matched-control result files.

The MPI component has its own [README](cluster_mpi/README.md), configuration examples, launcher-plan checks and numerical comparison tests. Set site-specific hosts in a private configuration copied from an example. The [microchunk component guide](agent_exchange/09_memory_boundary/T09_20261101_production_microchunk_regression/README.md) identifies its frozen first-segment regression, reference-data requirements, and historical device/controller restrictions. The public build compiles that source independently of the archived queue state. A new regression run needs its own declared hardware, references, and resource configuration.
