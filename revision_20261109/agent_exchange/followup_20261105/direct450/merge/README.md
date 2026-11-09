# Final direct 450-day merge

`merge_direct450.py` accepts only completed lane assignments that jointly cover days 1–450 exactly once. It checks each day receipt for all 10 segments and 1,954,680 half-baselines, the pinned scientific protocol fingerprint, the lane input identities, the local array dimensions and encoding, and the SHA256 of every array as it streams it. Daily receipt hashes are retained attestations from the lane runner. The merge rehashes the final lane accumulators. Those arrays store FP64 sums of daily FP32 numerators and FP64 weights. The script adds the lanes in order of their first assigned day, then divides once and writes FP32 `direct_C.f32` for the original NEST output grid. This operation changes the floating-point association relative to a chronological merge of individual days.

The input argument identifies the frozen sky. The STS arguments identify the two archived T10 arrays and their verified archive manifest. The direct and STS arms use their own denominators. Their comparison is at the same documented sky and geometry configuration with independently generated FP32 visibilities. The historical run retained its cumulative C/W and did not log the original per-segment visibility arrays.

Stage each lane's final `CURRENT.json` and the one generation directory it names. Preserve their relative paths. A direct committed `manifest.json` path can also be supplied. Each final lane needs about 3 GiB for `N.f64` and `W.f64`. The eight completed 2080 Ti lanes use about 24 GiB. The final comparison is run on that host, with the historical STS C/W arrays staged into the same dedicated workspace. The merge opens each lane once and reads bounded pixel chunks. Arrays may also reside in separate readable directories on the merge host. No full map is allocated in RAM.

Run from a host with Python 3.9 or later and NumPy. Ship `merge_direct450.py` and `pair_stats.py` together. The output directory must be new.

```bash
python3 -B merge_direct450.py \
  --lanes /staged/lane1 /staged/lane2 /staged/lane3 \
  --input /archive/B_10M.bin \
  --sts /archive/C1_450day10M.bin \
  --sts-w /archive/W1_450day10M.bin \
  --sts-manifest /archive/archive_verification.json \
  --output /staged/merged_direct450
```

The default output retains merged FP64 `direct_N.f64`, FP64 `direct_W.f64`, and FP32 `direct_C.f32`, about 3.75 GiB in total. `--pooled-only` keeps only the final C array and JSON. All arrays and `merge_summary.json` become visible in the final directory after verification. Check failed-run diagnostics before retrying. Preserve lane generations until their hashes and the output summary have been checked.

The JSON records full-domain and native-latitude metrics for direct and STS responses versus the frozen input sky, and for STS versus the pooled direct response. The separate raw direct N comparison retains its scale diagnostic. Report its output units separately from the normalized comparison. Relative L2 and RMSE divided by reference standard deviation use unaligned outputs. Affine fits are separate diagnostics. Each fit uses all pixels and then applies the same coefficients in the three regions. By default the script classifies analytic HEALPix NEST latitudes. To use the saved GPU-native classification, supply both `--regions` and `--regions-sha` with the aligned uint8 tags and their SHA256. The native coordinate is distinct from Galactic latitude.
