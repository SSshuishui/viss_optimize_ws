# Artifact manifest

| Manuscript evidence | Code or script | Recorded result |
|---|---|---|
| Direct 3D+DCF reference | `source/direct_3d_reference` | `results/controlled_method_comparison` |
| Exact-group and representative-group STS | `source/final_sts` | `results/controlled_method_comparison` and `results/scaling_and_orbital` |
| Five sampled orbital days | `experiments/run_five_day_paired_10m.sh` | `results/scaling_and_orbital/4090_final_suite_20260705_143116` |
| 1/2/4-GPU strong scaling | `experiments/run_scaling_10m.sh` | `results/summary/scaling_10mhz.csv` |
| Segment and grid-cell sensitivity | `source/final_sts/scripts` | `results/sensitivity` |
| Program and Nsight profiling | `source/final_sts/scripts/profile_10mhz_nsys.sh` | `results/sensitivity/profiling` |
| 30 MHz slab-size sensitivity | `source/stream_30mhz/scripts` | `results/slab_30mhz` |
| Sensitivity figure | `figures/plot_sensitivity_tradeoff.py` | `figures/sensitivity_tradeoff.png` |

Large binary inputs and reconstruction arrays are listed in `DATA.md` and are
not versioned. Generated executables and profiler databases are ignored.


The major-revision implementation and measured-software identities are mapped below. Paths in this table are relative to `revision_20261109`. The original tables above retain their original batches and software identities.

| Revision experiment | Code path | Public build component |
|---|---|---|
| Production visibility/cache replay | `agent_exchange/01_common_operator/T01_20261101_production_replay` | `direct450` builds its UUID-selected forward exporter |
| Mature optimized direct 3D and controlled direct | `agent_exchange/07_performance_system/T07_20261101_legacy_inverse_comparison` | `mature_direct` |
| Same-predicate corrected DCF numerator and denominator | `agent_exchange/01_common_operator/T01_20261101_full_corrected_dcf_reference` | `corrected_dcf` |
| Five selected full-day inverse workloads, four pairs per day | `agent_exchange/followup_20261105/deploy/five_day_timing` | `five_day` |
| Three repeated segmentation/cell-scale settings | `agent_exchange/followup_20261105/deploy/sweep_repeats` | `sweep` |
| Matched tile-path and device-ready geometry controls | `agent_exchange/followup_20261105/deploy/forward_ablation` | `forward` |
| Daily corrected direct N/W and transactional accumulated reference | `agent_exchange/followup_20261105/direct450` | `direct450` |
| Host/GPU direct, CPU FINUFFT and exact-mask cuFINUFFT controls | `agent_exchange/05_baselines/{T05_20261101_matched,T22_20261101_gpu_roi_v2,T22_20261101_masked_roi}` | `cpu_nufft` compiles the CPU direct libraries |
| Single-launch production NCU observations and bounded numerical gate | `agent_exchange/07_performance_system/T04_20261101_smallgate` | Archived profiling and analysis scripts |
| Full-field fixed-microchunk production regression | `agent_exchange/09_memory_boundary/T09_20261101_production_microchunk_regression` | `microchunk` |
| Static pixel ownership, flat/hierarchical host-buffer MPI | `cluster_mpi` | `mpi` |
| Current evaluation-server production/stream source | `server_snapshot` | Separate source identities and build instructions |

`revision_20261109/PUBLIC_SOURCE_FILES.json` records the workspace-relative origin, size and original SHA-256 for each copied file. An adapted file also records its published SHA-256 and the exact portability change. The MPI and microchunk components provide their own `PUBLIC_SOURCE_FILES.json` records with original and target hashes. Numerical CUDA sources retain their frozen contents.

`revision_20261109/provenance/SERVER_SOURCE_AUDIT.json` records the read-only comparison of deployed evaluation sources with the frozen originals. `server_snapshot/SOURCE_IDENTITY.json` identifies the separately retrieved current server sources. A source snapshot retains its own producing identity. The revision source validator checks manifest fingerprints and local quoted-header closure.

`results/revision_20261109` contains compact revision records. Every timing/result retains its scientific output definition and resource window. The five-day inverse ratio concerns direct DCF raw N and original STS C. The matched forward ratio covers its specified 1280-timestamp single-card kernel window. The 450-day reference pools daily N and W before normalization. Historical STS per-segment visibility hashes were unavailable, as documented with the comparison.
