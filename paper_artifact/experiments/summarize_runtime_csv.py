#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--row-type", default="day")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["row_type"] == args.row_type].copy()
    if df.empty:
        print("No rows")
        return
    cols = [
        "run_label", "btag", "nside", "day", "gpus_visible", "viss_mode", "recon_mode",
        "segs", "viss_s", "recon_s", "recon_total_s", "write_s", "total_day_s", "out_dir"
    ]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].to_string(index=False))

    if "gpus_visible" in df.columns and "recon_total_s" in df.columns and len(df) > 1:
        base = df.sort_values("gpus_visible").iloc[0]
        t1 = float(base["recon_total_s"])
        g1 = float(base["gpus_visible"])
        out = df[["run_label", "gpus_visible", "recon_total_s"]].copy()
        out["speedup_vs_first"] = t1 / out["recon_total_s"].astype(float)
        out["parallel_efficiency"] = out["speedup_vs_first"] / (out["gpus_visible"].astype(float) / g1)
        print("\nScaling table:")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
