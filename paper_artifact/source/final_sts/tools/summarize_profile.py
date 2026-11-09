#!/usr/bin/env python3
"""Summarize one or more profile CSV files emitted by the final-method binary."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

NUMERIC_SUM = [
    "visibility_dispatch_wall_s",
    "visibility_kernel_gpu_max_s",
    "visibility_d2h_gpu_max_s",
    "visibility_host_reduce_phase_s",
    "visibility_h2d_gpu_max_s",
    "group_build_wall_s",
    "reconstruction_stage_wall_s",
    "reconstruction_kernel_gpu_max_s",
    "segment_wall_s",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("profiles", nargs="+", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = []
    for path in args.profiles:
        with path.open(newline="") as f:
            data = [r for r in csv.DictReader(f) if r.get("day", "").isdigit()]
        if not data:
            continue
        out = {"profile": str(path), "segments": len(data)}
        for key in NUMERIC_SUM:
            out[key] = sum(float(r[key]) for r in data)
        out["mean_fit_rmse_wavelength"] = sum(float(r["fit_rmse_wavelength"]) for r in data) / len(data)
        out["max_fit_abs_wavelength"] = max(float(r["fit_max_abs_wavelength"]) for r in data)
        out["mean_nuniq"] = sum(float(r["nuniq"]) for r in data) / len(data)
        out["mean_group_size"] = sum(float(r["mean_group_size"]) for r in data) / len(data)
        out["max_group_size"] = max(int(float(r["max_group_size"])) for r in data)
        out["peak_gpu_memory_mib"] = max(float(r["peak_gpu_memory_mib"]) for r in data)
        out["grid_cell_scale_requested"] = data[0]["grid_cell_scale_requested"]
        rows.append(out)

    if not rows:
        raise SystemExit("No segment rows found")
    fields = list(rows[0].keys())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader(); w.writerows(rows)
    w = csv.DictWriter(__import__("sys").stdout, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

if __name__ == "__main__":
    main()
