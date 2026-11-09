#!/usr/bin/env python3
"""Check MPI output ownership, raw accumulators, normalization and references.

This verifies preparation smoke fixtures; its tolerances are not paper-quality
or scientific-recovery acceptance criteria. No GPU or remote connection used.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


class VerificationError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scalar(meta, key, *aliases):
    for node in (meta.get("config", {}), meta):
        for candidate in (key, *aliases):
            if candidate in node:
                return node[candidate]
    raise VerificationError(f"metrics missing required field {key}")


def load_array(path, dtype, count):
    path = Path(path)
    require(path.is_file(), f"missing array {path}")
    require(path.stat().st_size == count * np.dtype(dtype).itemsize,
            f"wrong array byte count: {path}")
    data = np.fromfile(path, dtype=dtype)
    if data.dtype.kind == "f":
        require(np.isfinite(data).all(), f"nonfinite values: {path}")
    return data


def bitwise(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def errors(candidate, reference):
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    diff = c - r
    refnorm = float(np.linalg.norm(r))
    peak = float(np.max(np.abs(r), initial=0))
    errnorm = float(np.linalg.norm(diff))
    maxabs = float(np.max(np.abs(diff), initial=0))
    # An all-zero reference is a strict zero check, not division by an arbitrary
    # floor that could make missing signal look like a successful comparison.
    return {
        "relative_l2": errnorm / refnorm if refnorm else (0.0 if errnorm == 0 else None),
        "absolute_rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_absolute": maxabs,
        "max_absolute_over_reference_peak": maxabs / peak if peak else (0.0 if maxabs == 0 else None),
        "reference_l2": refnorm,
        "reference_peak": peak,
        "bitwise_equal": bitwise(candidate, reference),
    }


def check_error(row, reltol, peaktol, label):
    require(row["relative_l2"] is not None and row["relative_l2"] <= reltol,
            f"{label}: relative L2 exceeds {reltol}: {row}")
    require(row["max_absolute_over_reference_peak"] is not None
            and row["max_absolute_over_reference_peak"] <= peaktol,
            f"{label}: peak-normalized max error exceeds {peaktol}: {row}")


def load_run(path, require_serial=True):
    root = Path(path).resolve()
    metrics_path = root / "metrics.json"
    require(metrics_path.is_file(), f"missing {metrics_path}")
    meta = json.loads(metrics_path.read_text())
    require(meta.get("status") == "completed", f"run not completed: {root}")
    nside = int(scalar(meta, "nside"))
    npix = int(scalar(meta, "npix"))
    ranks = int(scalar(meta, "ranks", "mpi_world_size", "world_size"))
    tile_pix = int(scalar(meta, "tile_pix"))
    require(nside > 0 and (nside & (nside - 1)) == 0, "nside is not a positive power of two")
    require(npix == 12 * nside * nside, "npix/nside mismatch")
    require(tile_pix == 256, "this validation protocol requires native 256-pixel tiles")
    require(ranks > 0, "invalid MPI rank count")

    with (root / "ownership.csv").open(newline="") as f:
        ownership = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
    require(len(ownership) == ranks, "ownership row count/rank count mismatch")
    cursor = 0
    tile_cursor = 0
    for expected_rank, row in enumerate(ownership):
        require(row["rank"] == expected_rank, "ownership ranks not in MPI rank order")
        require(row["pixel_start"] == cursor, "overlap, gap, or reordered pixel ownership")
        require(row["tile_start"] == tile_cursor, "overlap or gap in tile ownership")
        require(row["pixel_start"] == row["tile_start"] * tile_pix, "ownership starts inside native tile")
        require(row["pixel_count"] > 0 and row["tile_count"] > 0, "empty-rank ownership")
        expected_count = min(row["tile_count"] * tile_pix, npix - cursor)
        require(row["pixel_count"] == expected_count, "pixel/tile ownership count mismatch")
        cursor += row["pixel_count"]
        tile_cursor += row["tile_count"]
    require(cursor == npix, "ownership does not cover the full declared sky")
    require(tile_cursor == (npix + tile_pix - 1) // tile_pix, "incorrect native tile coverage")

    serial = meta.get("serial_check", {})
    if require_serial:
        for key in ("requested", "passed", "visibility_bitwise", "numerator_bitwise", "weight_bitwise", "map_bitwise"):
            require(serial.get(key) is True, f"same-partition serial gate {key} failed or absent: {root}")

    days = meta.get("days")
    require(isinstance(days, list) and len(days) > 0, "missing day output records")
    arrays = {}
    checks = []
    day_ids = []
    for day in days:
        dayid = int(day["day"])
        require(dayid not in arrays, "duplicate output day")
        day_ids.append(dayid)
        rel = Path(day.get("dir", f"day{dayid:04d}"))
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe day output path")
        daydir = root / rel
        files = {"numerator": ("numerator_f32.bin", "<f4"),
                 "weight": ("weight_u32.bin", "<u4"),
                 "map": ("map_f32.bin", "<f4")}
        full = {key: load_array(daydir / name, dtype, npix)
                for key, (name, dtype) in files.items()}
        for key, (name, dtype) in files.items():
            shards = [load_array(daydir / f"rank{row['rank']:04d}" / name, dtype, row["pixel_count"])
                      for row in ownership]
            joined = np.concatenate(shards)
            require(bitwise(joined, full[key]), f"MPI gather/order mismatch in {daydir}/{name}")
        num, weight, output = full["numerator"], full["weight"], full["map"]
        valid = weight > 0
        require(np.all(output[~valid] == 0), "unobserved pixels must normalize to zero")
        require(np.all(num[~valid] == 0), "nonzero numerator at zero weight")
        expected = np.zeros(npix, dtype=np.float32)
        expected[valid] = num[valid] / weight[valid].astype(np.float32)
        ulp = np.abs(np.spacing(np.abs(expected)))
        tolerance = 4 * ulp.astype(np.float64)
        diff = np.abs(output.astype(np.float64) - expected.astype(np.float64))
        require(np.all(diff <= tolerance), "map is not numerator/weight within 4 FP32 ulps")
        require(np.any(valid), "all pixels unobserved: smoke fixture is non-informative")
        require(np.linalg.norm(num.astype(np.float64)) > 0, "all-zero numerator: smoke fixture is non-informative")
        arrays[dayid] = full
        checks.append({"day": dayid, "valid_pixels": int(np.count_nonzero(valid)),
                       "gather_all_arrays_bitwise": True, "normalization_within_four_fp32_ulps": True,
                       "max_normalization_absolute_difference": float(diff.max(initial=0)),
                       "arrays_sha256": {key: sha256(daydir / name) for key, (name, _) in files.items()}})
    strategy = meta.get("config", {}).get("strategy", "flat")
    require(strategy in ("flat", "hierarchical"), "unknown reduction strategy")
    return {"root": str(root), "meta": meta, "nside": nside, "npix": npix, "ranks": ranks, "strategy": strategy,
            "ownership": ownership, "days": day_ids, "arrays": arrays,
            "checks": checks, "metrics_sha256": sha256(metrics_path)}


def compare(paths, original_map=None, reltol=2e-4, peaktol=2e-4,
            original_reltol=2e-5, original_peaktol=2e-5, require_serial=True,
            allow_strategy_difference=False):
    runs = [load_run(p, require_serial=require_serial) for p in paths]
    require(len(runs) > 0, "no input runs")
    reference = runs[0]
    report = {"schema": "cluster_mpi_preparation_comparison_v1", "status": "passed",
              "cross_node_test": False, "performance_claim": False,
              "cross_strategy_comparison_explicitly_authorized": allow_strategy_difference,
              "qualification": "Same-host preparation checks; numerical tolerances are integration guards, not scientific accuracy budgets.",
              "tolerances": {"partition_change_relative_l2": reltol, "partition_change_peak_normalized_max": peaktol,
                             "original_reference_relative_l2": original_reltol,
                             "original_reference_peak_normalized_max": original_peaktol},
              "runs": [], "partition_comparisons": [], "original_reference": []}
    for run in runs:
        require(run["nside"] == reference["nside"] and run["npix"] == reference["npix"], "different sky shape")
        require(run["days"] == reference["days"], "different output day IDs")
        require(allow_strategy_difference or run["strategy"] == reference["strategy"],
                "different reduction strategies require explicit cross-strategy comparison")
        require(run["meta"].get("input_fingerprint_fnv1a64") is not None, "missing input consistency fingerprint")
        require(run["meta"]["input_fingerprint_fnv1a64"] == reference["meta"].get("input_fingerprint_fnv1a64"),
                "different input fingerprints: refusing numerical comparison")
        for key in ("recon_mode", "segs", "time_start", "time_count", "grid_cell_scale", "orbit_seed",
                    "day_start", "day_count", "tlen_limit", "segments_limit", "btag", "lambda_m",
                    "sky_kind", "input", "synthetic"):
            a, b = scalar(run["meta"], key), scalar(reference["meta"], key)
            require(a == b, f"different config {key}: refusing numerical comparison")
        report["runs"].append({key: run[key] for key in ("root", "nside", "npix", "ranks", "strategy", "ownership", "days", "checks", "metrics_sha256")})
        if run is reference:
            continue
        for dayid in run["days"]:
            require(bitwise(run["arrays"][dayid]["weight"], reference["arrays"][dayid]["weight"]),
                    "weight differs across MPI rank counts")
            entry = {"candidate_ranks": run["ranks"], "reference_ranks": reference["ranks"], "day": dayid,
                     "candidate_strategy": run["strategy"], "reference_strategy": reference["strategy"],
                     "weight_bitwise_equal": True}
            for key in ("numerator", "map"):
                row = errors(run["arrays"][dayid][key], reference["arrays"][dayid][key])
                check_error(row, reltol, peaktol, f"rank comparison day {dayid} {key}")
                entry[key] = row
            report["partition_comparisons"].append(entry)
    if original_map is not None:
        require(reference["ranks"] == 1, "original reference must be compared to one-rank candidate first")
        require(len(reference["days"]) == 1, "--original-map requires one output day")
        original = load_array(original_map, "<f4", reference["npix"])
        require(np.linalg.norm(original.astype(np.float64)) > 0, "all-zero original reference")
        row = errors(reference["arrays"][reference["days"][0]]["map"], original)
        check_error(row, original_reltol, original_peaktol, "immutable original pipeline regression")
        report["original_reference"].append({"path": str(Path(original_map).resolve()), "sha256": sha256(original_map), **row})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--original-map", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relative-l2-tolerance", type=float, default=2e-4)
    parser.add_argument("--peak-error-tolerance", type=float, default=2e-4)
    parser.add_argument("--original-relative-l2-tolerance", type=float, default=2e-5)
    parser.add_argument("--original-peak-error-tolerance", type=float, default=2e-5)
    parser.add_argument("--no-require-serial", action="store_true")
    parser.add_argument("--allow-strategy-difference", action="store_true",
                        help="Explicitly authorize comparison of flat and hierarchical reductions")
    args = parser.parse_args()
    try:
        for value in (args.relative_l2_tolerance, args.peak_error_tolerance,
                      args.original_relative_l2_tolerance, args.original_peak_error_tolerance):
            require(math.isfinite(value) and value >= 0, "invalid tolerance")
        report = compare(args.runs, args.original_map, args.relative_l2_tolerance,
                         args.peak_error_tolerance, args.original_relative_l2_tolerance,
                         args.original_peak_error_tolerance, not args.no_require_serial,
                         args.allow_strategy_difference)
        exit_code = 0
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        report = {"schema": "cluster_mpi_preparation_comparison_v1", "status": "failed",
                  "failure": str(exc), "cross_node_test": False, "performance_claim": False}
        exit_code = 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "report": str(args.output),
                      "failure": report.get("failure")}, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
