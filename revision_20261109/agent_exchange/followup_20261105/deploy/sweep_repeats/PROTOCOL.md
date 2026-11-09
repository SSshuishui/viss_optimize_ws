# R2.11 representative sweep timing

Freeze three day-1 STS configurations before timing: (segments, grid scale)
(10, 1), (20, 1), and (10, 1.5). Use all 201,326,592 NEST output pixels
at Nside 4096 and 1,954,680 half-baselines from the original ten consumed-n
G1 visibility cache files. Each case uses every input record. The 20-segment
case splits the concatenated payload by timestamps using the sampled replay's
floor-plus-remainder rule, with ten 3491-timestamp segments followed by ten
3490-timestamp segments. Preserve the row bytes and order. Preflight records
SHA256 of the original payload and each case's segment slices. No new
visibility generation, extra cache copies, or full maps are required.

Use one RTX 4090 selected by UUID and one binary with the frozen CUDA
headers, CUDA 12.4 sm_89 and the archived fast-math setting. Use original
midpoint representatives, grid phase, cell weighting and normalization.
The baseline-case inverse timer uses the same boundary as the archived
full-domain C replay. It includes per-segment H2D, fit, grid/group,
representative gathering, reconstruction and accumulation. Cache reading,
CPU partitioning and decode, allocation and coordinates, final normalization,
and output scans are outside the inverse timer. Record those times separately.

Run one complete numerical warm-up per case, then three formal blocks:

1. (10,1), (20,1), (10,1.5)
2. (20,1), (10,1.5), (10,1)
3. (10,1.5), (10,1), (20,1)

The warm-up also serves as the numerical gate. In every run verify actual
segment offsets and payload SHAs against preflight. The per-segment occupied
cell count, grid interval count and fit match the corresponding archived
sampled replay, allowing only decimal CSV precision for the floating
metadata. Require the same 4096 sample IDs, exact W, and both relative L2
and max absolute error over reference peak at most 1e-6 for raw N. All
201,326,592 N/W/C entries must be finite and their full hashes must match
the corresponding accepted warm-up within a case. The three expected total
group counts are 376167, 376611 and 257440.

Keep all valid formal times. Report the three individual values, median,
sample standard deviation and min/max. This study estimates single-card
full-output inverse timing variability for three representative settings.
The historical four-card forward-plus-inverse sweep and its other settings
keep their original timing scope and repetition counts. No inference about
450-day accumulation or held output accuracy across different settings.
