#!/usr/bin/env python3
"""Check early and late regenerated visibility and an independent full-day inverse."""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np

PIXELS = 12 * 4096**2
HALF = 195468


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def publish(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def idle(uuid):
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", uuid, "--query-gpu=uuid,memory.free",
         "--format=csv,noheader,nounits"], text=True
    ).strip().split(",")
    if len(raw) != 2 or raw[0].strip() != uuid or int(raw[1]) < 8500:
        raise RuntimeError("Target GPU is absent or below the memory guard")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if any(line.split(",")[0].strip() == uuid for line in processes.splitlines()):
        raise RuntimeError(f"Target GPU already has a compute context: {uuid}")


def launch(command, log, uuid, cwd):
    idle(uuid)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=uuid,
               OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    start = time.monotonic()
    with log.open("xb") as output:
        subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT,
                       timeout=1800, check=True)
    idle(uuid)
    return time.monotonic() - start


def error(candidate, reference):
    new = np.asarray(candidate, dtype=np.float64)
    old = np.asarray(reference, dtype=np.float64)
    if new.shape != old.shape or not (np.isfinite(new).all() and np.isfinite(old).all()):
        raise ValueError("Non-finite or mismatched arrays")
    delta = new - old
    return {
        "rel_l2": float(np.sqrt(np.sum(delta * delta) / np.sum(old * old))),
        "max_over_peak": float(np.max(np.abs(delta)) / np.max(np.abs(old))),
        "changed_values": int(np.count_nonzero(delta)),
    }


def within(metrics, limit):
    return all(metrics[key] <= limit for key in ("rel_l2", "max_over_peak"))


def compare_cache(new, old, identity):
    if old.stat().st_size != identity["bytes"] or sha256(old) != identity["sha256"]:
        raise ValueError(f"Frozen reference failed hash verification: {old}")
    if new.stat().st_size != identity["bytes"]:
        raise ValueError(f"Wrong cache size: {new}")
    expected_header = (b"PREPLAY1", *identity["header"])
    for path in (old, new):
        with path.open("rb") as stream:
            if struct.unpack("<8s7i", stream.read(36)) != expected_header:
                raise ValueError(f"Wrong cache header: {path}")
    a, b = (np.memmap(path, dtype="<f4", mode="r", offset=36, shape=(HALF, 15))
            for path in (new, old))
    metrics = {
        "uvw": error(a[:, :3], b[:, :3]),
        "endpoints": error(a[:, [3, 4, 5, 7, 8, 9]], b[:, [3, 4, 5, 7, 8, 9]]),
        "inverse_norms": error(a[:, [6, 10]], b[:, [6, 10]]),
        "raw_visibility": error(a[:, 11:13], b[:, 11:13]),
        "consumed_visibility": error(a[:, 13:], b[:, 13:]),
    }
    limits = dict(uvw=3e-6, endpoints=3e-6, inverse_norms=3e-6,
                  raw_visibility=3e-3, consumed_visibility=3e-3)
    return {"cache": new.name, "sha256": sha256(new), "errors": metrics, "limits": limits,
            "passed": all(within(metrics[key], limit) for key, limit in limits.items())}


def forward(base, work, uuid):
    frozen = base / "gate_inputs"
    identities = {Path(row["path"]).name: row for row in
                  json.loads((base / "gate_inputs.json").read_text())}
    work.mkdir(exist_ok=False)
    (work / "cache").mkdir()
    report = {"gpu_uuid": uuid, "forward_sha256": sha256(base / "bin/forward_uuid"), "dates": {}}
    for day, segments, time_count in ((1, 10, 0), (450, 1, 6981)):
        command = [str(base / "bin/forward_uuid"), f"--gpus={uuid}", "--btag=10M",
                   "--nside=4096", f"--day_start={day}", "--day_count=1", f"--segs={segments}",
                   "--time_start=0", f"--time_count={time_count}", "--orbit_seed=42",
                   "--gen_gpu_index=0", "--blockage=1", "--occ_mode=2", "--viss_tile_pix=256",
                   "--grid_cell_scale=1", "--B_mode=bin", "--C_mode=bin", "--profile=0",
                   "--cache_only=1", "--write_output=0", "--write_viss=0", "--write_baseline_txt=0",
                   f"--sky_dir={base / 'inputs/earth_10Mhz'}", f"--out_dir={work}",
                   f"--cache_root={work / 'cache'}"]
        seconds = launch(command, work / f"forward_day{day}.log", uuid, work)
        comparisons = []
        for segment in range(1, segments + 1):
            name = f"day{day}_seg{segment}.bin"
            comparisons.append(compare_cache(work / "cache" / name, frozen / name, identities[name]))
        report["dates"][str(day)] = {"argv": command, "wall_s": seconds, "segments": comparisons}
        publish(work / "forward_gate.json", report)
        if not all(record["passed"] for record in comparisons):
            raise RuntimeError(f"Day {day} visibility gate exceeded the declared tolerances")
    report["passed"] = True
    publish(work / "forward_gate.json", report)


def direct(base, work, uuid):
    saved = json.loads((work / "forward_gate.json").read_text())
    if saved.get("passed") is not True or saved["gpu_uuid"] != uuid:
        raise RuntimeError("Complete the forward gate on this UUID first")
    out = work / "direct"
    out.mkdir(exist_ok=False)
    dcf = base / "inputs/dcf_mb_10M_days450_seed42.bin"
    command = [str(base / "bin/direct_daily"), f"--gpus={uuid}", "--nside=4096", "--day=1",
               "--max_segments=10", f"--dcf_bin={dcf}", f"--cache_root={work / 'cache'}",
               f"--out_dir={out}"]
    seconds = launch(command, work / "direct_day1.log", uuid, work)
    ref = base / "gate_inputs"
    ids = np.fromfile(ref / "sample_ids.u32", dtype="<u4")
    if ids.size != 8192 or np.unique(ids).size != 8192 or np.any(ids >= PIXELS):
        raise ValueError("Invalid fixed sample directions")
    stats = json.loads((out / "full_stats.json").read_text())
    report = {"gpu_uuid": uuid, "argv": command, "wall_s": seconds, "full_stats": stats,
              "direct_sha256": sha256(base / "bin/direct_daily"), "sample_count": int(ids.size)}
    for key, filename, fmt, name, limit in (
        ("rawN", "raw_N.f32", "<f4", "sample_N.f32", 1e-3),
        ("W", "W.f64", "<f8", "sample_W.f64", 1e-5),
    ):
        path = out / filename
        if path.stat().st_size != PIXELS * np.dtype(fmt).itemsize:
            raise ValueError(f"Unexpected full image shape: {path}")
        old = np.fromfile(ref / name, dtype=fmt)
        new = np.memmap(path, dtype=fmt, mode="r", shape=(PIXELS,))
        report[key] = {"errors": error(new[ids], old), "limit": limit,
                       "full_hash_matches": sha256(path) == stats[key]["sha256"]}
        del new
    report["passed"] = all(record["full_hash_matches"] and within(record["errors"], record["limit"])
                           for record in (report["rawN"], report["W"]))
    publish(work / "direct_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("Direct day-1 gate exceeded its declared tolerances")
    for path in (out / "raw_N.f32", out / "W.f64"):
        path.unlink()
    shutil.rmtree(work / "cache")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("forward", "direct"))
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--gpu-uuid", required=True)
    args = parser.parse_args()
    base = args.base.resolve()
    {"forward": forward, "direct": direct}[args.phase](base, base / "portability_gate", args.gpu_uuid)
