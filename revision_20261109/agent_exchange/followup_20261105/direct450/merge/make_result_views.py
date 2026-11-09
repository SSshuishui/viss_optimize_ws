#!/usr/bin/env python3
"""Save small whole-field previews and fixed-direction audit data after merge."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def parent_means(path, expected_sha):
    digest = hashlib.sha256()
    means = []
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 * (1 << 20)), b""):
            digest.update(data)
            values = np.frombuffer(data, dtype="<f4")
            if values.size % 1024 or not np.isfinite(values).all():
                raise ValueError(f"Invalid preview input: {path}")
            means.append(values.reshape(-1, 1024).mean(axis=1, dtype=np.float64).astype("<f4"))
    if digest.hexdigest() != expected_sha:
        raise ValueError(f"Preview SHA mismatch: {path}")
    result = np.concatenate(means)
    if result.size != 12 * 128**2:
        raise ValueError("Wrong preview pixel count")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    result = base / "merged_direct450_20261109"
    summary = json.loads((result / "merge_summary.json").read_text())
    if summary["state"] != "completed":
        raise ValueError("Merge is incomplete")
    history = base / "history_sts_450_20261109"
    paths = {
        "input": base / "inputs/earth_10Mhz/B_10M.bin",
        "direct": result / "direct_C.f32",
        "sts": history / "C1_450day10M.bin",
    }
    expected = {
        "input": summary["input_files"]["input"]["sha256"],
        "direct": summary["output_files"]["C"]["sha256"],
        "sts": summary["input_files"]["sts_C"]["sha256"],
    }
    preview = {key: parent_means(path, expected[key]) for key, path in paths.items()}
    pair = summary["comparisons"]["sts_vs_dcf_pooled"]
    preview.update(alpha=np.array(pair["global_alpha"]), beta=np.array(pair["global_beta"]),
                   preview_nside=np.array(128), full_nside=np.array(4096))
    preview_file = result / "preview_nside128.npz"
    if preview_file.exists():
        raise ValueError("Preview already exists")
    np.savez_compressed(preview_file, **preview)

    ids = np.fromfile(base / "gate_inputs/sample_ids.u32", dtype="<u4")
    if ids.size != 8192 or np.unique(ids).size != ids.size:
        raise ValueError("Invalid fixed IDs")
    npix = 12 * 4096**2
    sample = {"ids": ids}
    for name, path, dtype in (
        ("input_B", paths["input"], "<f4"),
        ("direct_N", result / "direct_N.f64", "<f8"),
        ("direct_W", result / "direct_W.f64", "<f8"),
        ("direct_C", paths["direct"], "<f4"),
        ("sts_C", paths["sts"], "<f4"),
        ("sts_W", history / "W1_450day10M.bin", "<u4"),
        ("regions", history / "native_regions.u8", "u1"),
    ):
        values = np.memmap(path, dtype=dtype, mode="r", shape=(npix,))
        sample[name] = values[ids].copy()
        del values
    if not np.array_equal(sample["direct_C"], (sample["direct_N"] / sample["direct_W"]).astype("<f4")):
        raise ValueError("Sample pooled output disagrees with N/W")
    sample_file = result / "audit_samples8192.npz"
    if sample_file.exists():
        raise ValueError("Sample output already exists")
    np.savez_compressed(sample_file, **sample)
    print(preview_file)
    print(sample_file)


if __name__ == "__main__":
    main()
