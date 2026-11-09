#!/usr/bin/env python3
"""Compare two headerless float32 dirty-map binaries without loading them fully."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--chunk-elements", type=int, default=8_000_000)
    p.add_argument("--json-out", type=Path)
    p.add_argument("--csv-out", type=Path)
    p.add_argument("--label", default="comparison")
    return p.parse_args()


def checked_length(path: Path) -> int:
    size = path.stat().st_size
    if size % 4:
        raise ValueError(f"{path} size {size} is not divisible by 4 bytes")
    return size // 4


def main() -> None:
    args = parse_args()
    n_ref = checked_length(args.reference)
    n_cand = checked_length(args.candidate)
    if n_ref != n_cand:
        raise ValueError(f"length mismatch: {n_ref} vs {n_cand}")
    if n_ref == 0:
        raise ValueError("empty map")

    x = np.memmap(args.reference, dtype=np.float32, mode="r", shape=(n_ref,))
    y = np.memmap(args.candidate, dtype=np.float32, mode="r", shape=(n_ref,))

    n = 0
    sx = sy = sx2 = sy2 = sxy = sse_raw = 0.0
    xmin = math.inf
    xmax = -math.inf

    for i in range(0, n_ref, args.chunk_elements):
        xx = np.asarray(x[i : i + args.chunk_elements], dtype=np.float64)
        yy = np.asarray(y[i : i + args.chunk_elements], dtype=np.float64)
        finite = np.isfinite(xx) & np.isfinite(yy)
        if not np.all(finite):
            xx = xx[finite]
            yy = yy[finite]
        if xx.size == 0:
            continue
        n += int(xx.size)
        sx += float(xx.sum(dtype=np.float64))
        sy += float(yy.sum(dtype=np.float64))
        sx2 += float(np.dot(xx, xx))
        sy2 += float(np.dot(yy, yy))
        sxy += float(np.dot(xx, yy))
        d = xx - yy
        sse_raw += float(np.dot(d, d))
        xmin = min(xmin, float(xx.min()))
        xmax = max(xmax, float(xx.max()))

    if n == 0:
        raise ValueError("no finite paired elements")

    mx = sx / n
    my = sy / n
    var_x_sum = max(0.0, sx2 - sx * sx / n)
    var_y_sum = max(0.0, sy2 - sy * sy / n)
    cov_sum = sxy - sx * sy / n
    pearson = cov_sum / math.sqrt(var_x_sum * var_y_sum) if var_x_sum > 0 and var_y_sum > 0 else math.nan

    alpha = cov_sum / var_y_sum if var_y_sum > 0 else 0.0
    beta = mx - alpha * my

    sse_affine = 0.0
    for i in range(0, n_ref, args.chunk_elements):
        xx = np.asarray(x[i : i + args.chunk_elements], dtype=np.float64)
        yy = np.asarray(y[i : i + args.chunk_elements], dtype=np.float64)
        finite = np.isfinite(xx) & np.isfinite(yy)
        if not np.all(finite):
            xx = xx[finite]
            yy = yy[finite]
        if xx.size == 0:
            continue
        d = xx - (alpha * yy + beta)
        sse_affine += float(np.dot(d, d))

    ref_range = xmax - xmin
    raw_rmse = math.sqrt(sse_raw / n)
    affine_rmse = math.sqrt(sse_affine / n)
    raw_nrmse = raw_rmse / ref_range if ref_range > 0 else math.nan
    affine_nrmse = affine_rmse / ref_range if ref_range > 0 else math.nan
    relative_l2 = math.sqrt(sse_raw / sx2) if sx2 > 0 else math.nan

    result = {
        "label": args.label,
        "reference": str(args.reference),
        "candidate": str(args.candidate),
        "paired_finite_elements": n,
        "reference_min": xmin,
        "reference_max": xmax,
        "reference_range": ref_range,
        "pearson": pearson,
        "affine_gain_alpha": alpha,
        "affine_offset_beta": beta,
        "raw_rmse": raw_rmse,
        "raw_nrmse_by_reference_range": raw_nrmse,
        "affine_rmse": affine_rmse,
        "affine_nrmse_by_reference_range": affine_nrmse,
        "relative_l2_error": relative_l2,
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        write_header = not args.csv_out.exists()
        with args.csv_out.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                w.writeheader()
            w.writerow(result)


if __name__ == "__main__":
    main()
