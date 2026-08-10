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

