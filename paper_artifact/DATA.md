# Input data and numerical identities

The measured production runs require externally supplied sky inputs with the following identities. Both files contain flat little-endian FP32 brightness values in HEALPix NEST order.

| Input | Frequency | Nside | Pixel count | Bytes | SHA-256 |
|---|---|---:|---:|---:|---|
| `B_10M.bin` | 10 MHz | 4096 | 201,326,592 | 805,306,368 | `277b6cb7299f2e8e54cc1758a045e7589686e28dc79a1df2f56aba5fc2d2d5ac` |
| `B_30M.bin` | 30 MHz | 16384 | 3,221,225,472 | 12,884,901,888 | `2114f0adf6d0f7d01362998101c15190ce0441aa9b008b912931b4fe5be27898` |

These inputs are excluded from the code artifact. Their upstream redistribution terms and generation histories must be respected when obtaining copies. The recorded hashes identify the actual experimental inputs. A newly generated sky model or a different interpolation workflow can produce a different array. The historical 30 MHz preparation includes a one-dimensional spline workflow from `B_1M.txt`. The artifact does not assert that a current sky-model package recreates either recorded byte sequence.

The compact radial DCF input is included in the original artifact at `source/direct_3d_reference/dcf_mb_10M_days450_seed42.bin`. It contains 80,064 bytes and has SHA-256 `8f5d7f206e5dc1d51282b5be43bfbc237b61d7d63a4665ee10997a244c7e4d2f`. Corrected direct references use the recorded 450-day, seed-42 DCF identity. Code paths that export raw numerator N retain that output scope. The daily direct exporter additionally produces a same-predicate FP64 denominator W.

Validate a supplied file before numerical reproduction. This streaming command avoids loading the full sky into host memory.

```bash
python3 - /path/to/B_10M.bin 277b6cb7299f2e8e54cc1758a045e7589686e28dc79a1df2f56aba5fc2d2d5ac <<'PY'
import hashlib
from pathlib import Path
import sys
path = Path(sys.argv[1])
state = hashlib.sha256()
with path.open('rb') as stream:
    for block in iter(lambda: stream.read(8 * 1024**2), b''):
        state.update(block)
assert state.hexdigest() == sys.argv[2], 'Input SHA-256 mismatch'
print(path.name, path.stat().st_size, state.hexdigest())
PY
```

Production replay caches use `PREPLAY1` files. The 36-byte little-endian header is `8s7i`, with magic, Nside, day, segment, first timestamp, timestamp count, half-baseline count, and forward GPU count. Each record contains fifteen FP32 values in the order `u,v,w,x1,y1,z1,inv1,x2,y2,z2,inv2,raw_re,raw_im,re,im`. The final complex pair is the consumed phase-corrected visibility. A ten-segment 10 MHz day contains 1,954,680 half-baseline records. Cache generation instructions are in the revision guide.

The historical five-day and sweep drivers require additional fixed sample IDs, accepted sample numerators/counts, and group metadata. Their paths, byte counts, and hashes are in each driver's `input_identities.json`. These derived arrays are excluded. Replaying an archived timing batch requires its accepted cache/reference set. A new hardware/input run should record a new dataset identity and validate its reference independently. Changing a manifest constitutes a new experiment configuration.

CPU and Fourier controls use externally supplied reduced geometry exports, NEST coordinate prefixes, and selected production-cache records. The included `export_existing_inputs.py` and `mask_statistics.py` reconstruct the declared mask cases from those inputs and retain SHA checks. Their source paths and command arguments are described in the revision guide. Compiling all implementations and running static validation requires none of these numerical arrays.

Generated visibility caches, N/W/C maps, lane checkpoints, sample binaries, compiled executables, shared libraries, and profiler databases are excluded from version control. Published results contain compact timings and numerical summaries. Deployment credentials, private SSH configuration, and machine-specific host lists are excluded.
