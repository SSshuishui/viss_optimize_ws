#!/usr/bin/env python3
"""Summarize the complete A/B run and retain every observation."""
import csv
import json
import math
from pathlib import Path
from statistics import median
import sys


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def timings(rows, names, value, mode):
    return {name: [float(row[value]) for row in rows if row[mode] == name] for name in names}


def main(out):
    full = read(out / "timings.csv")
    formal = [row for row in full if row["formal"] == "1"]
    assert len(full) == 10 and len(formal) == 8
    assert [formal[i]["order"] for i in range(0, 8, 2)] == ["AB", "BA", "BA", "AB"]
    kernels = timings(formal, ("tilecone", "point_only"), "kernel_s", "variant")
    geometry = read(out / "geometry_timings.csv")
    prep = timings(geometry, ("online_ready", "cached_file_to_device"), "wall_s", "mode")
    assert all(len(seq) == 4 for seq in kernels.values())
    assert all(len(seq) == 8 for seq in prep.values())
    assert all(math.isfinite(t) and t > 0 for seq in [*kernels.values(), *prep.values()] for t in seq)
    errors = read(out / "pair_errors.csv")
    assert len(errors) == 5 and all(
        math.isfinite(float(row[key])) and float(row[key]) >= 0
        for row in errors for key in ("relative_l2", "max_over_peak", "max_absolute")
    )
    ratios = [b / a for a, b in zip(kernels["tilecone"], kernels["point_only"])]
    summary = {
        "metadata": read(out / "metadata.csv")[0],
        "return_code": int((out / "exit_code.txt").read_text()),
        "kernel_s": kernels,
        "kernel_median_s": {name: median(seq) for name, seq in kernels.items()},
        "point_only_over_tilecone_median_ratio": median(kernels["point_only"]) / median(kernels["tilecone"]),
        "point_only_over_tilecone_paired_ratios": ratios,
        "point_only_over_tilecone_median_paired_ratio": median(ratios),
        "geometry_prepare_wall_s": prep,
        "geometry_prepare_median_s": {name: median(seq) for name, seq in prep.items()},
        "all_geometry_repeats_bitwise": all(row["byte_equal_to_snapshot"] == "1" for row in geometry),
        "all_kernel_repeats_bitwise_within_variant": all(row["repeat_bitwise"] == "1" for row in formal),
        "tolerance_relative_l2_and_max_over_peak": 1e-5,
        "all_pairs_numerically_close": all(
            float(row["relative_l2"]) <= 1e-5 and float(row["max_over_peak"]) <= 1e-5
            for row in errors
        ),
        "raw_complex_vector_pair_errors": errors,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
