#!/usr/bin/env python3
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def relative_l2(a,b):
    numerator=sum(abs(x-y)**2 for x,y in zip(a,b))
    denominator=sum(abs(y)**2 for y in b)
    return math.sqrt(numerator/denominator) if denominator else None

def main():
    path=Path(sys.argv[1]); output=Path(sys.argv[2])
    rows=list(csv.DictReader(path.open()))
    assert len(rows)==56
    complex_column=lambda a,b:[complex(float(r[a]),float(r[b])) for r in rows]
    full=complex_column("full_re","full_im")
    half=complex_column("half_or_conjugate_re","half_or_conjugate_im")
    oracle=complex_column("fp64_math_re","fp64_math_im")
    fixed=complex_column("fp64_fixed_float_phase_re","fp64_fixed_float_phase_im")
    counts=[int(r['visible_pixels']) for r in rows]
    assert any(c==256 for c in counts) and any(c==0 for c in counts) and any(0<c<256 for c in counts)
    assert all(counts[j]==counts[j+28] for j in range(28))
    full_half=relative_l2(full,half)
    same_phase=relative_l2(full,fixed)
    active=[i for i,z in enumerate(oracle) if abs(z)>1e-12]
    data={
        "fixture":"Controlled synthetic 1tile x 256 directions x 28 independent baseline pairs; not a production sky-image accuracy claim",
        "csv":str(path),"csv_sha256":digest(path),
        "n_directions":256,"n_full_baselines":56,"shared_arrays":5,
        "visible_count_min":min(counts),"visible_count_max":max(counts),
        "partial_mask_baselines":sum(0<c<256 for c in counts),
        "max_abs_phase_rad":max(float(r['max_abs_phase_rad']) for r in rows),
        "minimum_point_mask_margin":min(float(r['minimum_point_mask_margin']) for r in rows),
        "full_vs_folded_half_relative_l2":full_half,
        "full_vs_folded_half_max_absolute":max(abs(a-b) for a,b in zip(full,half)),
        "full_vs_independent_fp64_math_relative_l2":relative_l2(full,oracle),
        "full_vs_independent_fp64_math_max_absolute":max(abs(a-b) for a,b in zip(full,oracle)),
        "full_vs_independent_fp64_math_max_relative_nonzero":max(abs(full[i]-oracle[i])/abs(oracle[i]) for i in active),
        "per_baseline_relative_denominator_floor":1e-12,
        "oracle_nonzero_baselines":len(active),
        "full_vs_fp64_fixed_float_phase_relative_l2":same_phase,
        "acceptance":{
            "full_vs_folded_half_l2_limit":1e-6,
            "fixed_float_phase_fp64_l2_limit":1e-4,
            "pass":full_half<1e-6 and same_phase<1e-4,
            "math_oracle":"Reported, not accepted by relaxing a frozen threshold; float phase/trig/reduction errors remain visible",
        },
    }
    output.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data,indent=2))
    if not data['acceptance']['pass']:sys.exit(1)

if __name__=='__main__':main()
