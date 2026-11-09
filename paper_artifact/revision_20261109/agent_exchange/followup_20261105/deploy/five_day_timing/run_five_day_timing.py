#!/usr/bin/env python3
"""Run isolated, cached, full-output inverse comparisons on five selected days."""
import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import struct
import subprocess
import tempfile

TASK = Path(__file__).resolve().parent
DAYS = (1, 113, 225, 337, 450)
PAIRS = (("B", "C"), ("C", "B"), ("C", "B"), ("B", "C"))
BINARIES = {"B": "replay_legacy_combined_full", "C": "replay_sts_full"}
METHOD_IDS = {"B": "legacy_combined_fastpath", "C": "sts_grid"}
PIXELS = 201326592
GIB = 1024**3


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else digest_stream(handle)


def digest_stream(handle):
    state = hashlib.sha256()
    for block in iter(lambda: handle.read(1024**2), b""):
        state.update(block)
    return state.hexdigest()


def verify(path, record):
    if ("bytes" in record and path.stat().st_size != record["bytes"]) or digest(path) != record["sha256"]:
        raise RuntimeError(f"frozen input identity mismatch: {path}")


def records_under(folder):
    return [{"path": str(path.relative_to(folder)), "bytes": path.stat().st_size,
             "sha256": digest(path)} for path in sorted(folder.rglob("*"))
            if path.is_file() and path.name != "manifest.json"]


def nvidia_query(*query):
    result = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=20)
    return result.stdout.strip()


def boundary(run_dir, gpu_uuid, label):
    if (run_dir / "PAUSE").exists():
        raise RuntimeError("PAUSE marker present")
    usage = os.statvfs(run_dir)
    free_bytes = usage.f_bavail * usage.f_frsize
    local_bytes = sum(p.stat().st_size for p in run_dir.rglob("*") if p.is_file())
    gpu = nvidia_query("-i", gpu_uuid, "--query-gpu=uuid,name,memory.total,memory.free,driver_version")
    apps = nvidia_query("--query-compute-apps=gpu_uuid,pid")
    contexts = [line for line in apps.splitlines() if line.split(",")[0].strip() == gpu_uuid]
    write_json(run_dir / "boundaries" / f"{label}.json",
               {"at": utc_now(), "gpu": gpu, "target_contexts": contexts,
                "free_disk_bytes": free_bytes, "own_batch_bytes": local_bytes})
    if gpu.split(",")[0].strip() != gpu_uuid:
        raise RuntimeError("GPU inventory does not match the requested UUID")
    if contexts:
        raise RuntimeError(f"target GPU busy at {label}: {gpu_uuid}")
    if free_bytes < 100 * GIB or local_bytes >= 100 * 1024**2:
        raise RuntimeError(f"disk guard failed at {label}")


def verify_inputs(exchange_root, cache_root, dcf_path, manifest):
    seen = set()
    for record in manifest["caches"]:
        path = cache_root / Path(record["path"]).name
        verify(path, record)
        with path.open("rb") as handle:
            magic, *values = struct.unpack("<8s7i", handle.read(36))
        if magic != b"PREPLAY1" or values != record["header"]:
            raise RuntimeError(f"cache header mismatch: {path}")
        nside, day, segment, offset, timestamps, baselines, devices = values
        if (nside, offset, timestamps, baselines, devices) != (4096, 6981*(segment-1), 6981, 195468, 1):
            raise RuntimeError(f"cache workload mismatch: {path}")
        seen.add((day, segment))
    if seen != {(day, seg) for day in DAYS for seg in range(1, 11)} or len(manifest["caches"]) != 50:
        raise RuntimeError("cache manifest does not cover the frozen five-day matrix")
    verify(dcf_path, manifest["dcf"])
    for day in DAYS:
        for record in manifest["references"][str(day)].values():
            verify(exchange_root / record["path"], record)


def scientific_gate(folder, day, method, references, exchange_root):
    import numpy as np

    ids = np.fromfile(folder / "sample_ids.u32", dtype="<u4")
    expected_ids = np.fromfile(exchange_root / references["ids"]["path"], dtype="<u4")
    actual = np.fromfile(folder / "sample_N.f32", dtype="<f4")
    if ids.shape != (8192,) or not np.array_equal(ids, expected_ids) or actual.shape != ids.shape:
        raise RuntimeError("8192 frozen sample IDs or output sample dimensions failed")
    if not np.isfinite(actual).all():
        raise RuntimeError("nonfinite sampled raw numerator")
    if method == "B":
        expected = np.fromfile(exchange_root / references["B"]["path"], dtype="<f4")
        expected_weights = None
    else:
        with np.load(exchange_root / references["C"]["path"], allow_pickle=False) as accepted:
            if not np.array_equal(ids, accepted["pixel_ids"]):
                raise RuntimeError("STS reference pixel IDs differ")
            expected = np.asarray(accepted["rep_N"], dtype="<f4")
            expected_weights = np.asarray(accepted["rep_W"], dtype="<u4")
    if expected.shape != ids.shape or not np.isfinite(expected).all():
        raise RuntimeError("invalid frozen raw numerator reference")
    ref = expected.astype(np.float64)
    error = actual.astype(np.float64) - ref
    ref_l2 = float(np.linalg.norm(ref))
    ref_peak = float(np.max(np.abs(ref)))
    if ref_l2 <= 0 or ref_peak <= 0:
        raise RuntimeError("zero reference norm")
    relative_l2 = float(np.linalg.norm(error) / ref_l2)
    peak_relative = float(np.max(np.abs(error)) / ref_peak)
    # Compare the actual bits, including signed zeros.
    exact_n = np.array_equal(actual.view("<u4"), expected.view("<u4"))
    exact_w = None
    if method == "C":
        weights = np.fromfile(folder / "sample_W.u32", dtype="<u4")
        exact_w = np.array_equal(weights, expected_weights)
    full = read_json(folder / "full_stats.json")
    for name in (("rawN",) if method == "B" else ("rawN", "W", "C")):
        stats = full[name]
        if (stats["count"] != PIXELS or
                stats.get("finite_count", PIXELS - stats.get("nonfinite", 0)) != PIXELS or
                not re.fullmatch(r"[0-9a-f]{64}", stats["sha256"])):
            raise RuntimeError(f"invalid full-output scan for {name}")
    passed = (relative_l2 <= 1e-3 and peak_relative <= 1e-3) if method == "B" else (exact_n and exact_w)
    return {"day": day, "passed": bool(passed), "relative_l2": relative_l2,
            "max_abs_over_reference_peak": peak_relative, "sample_raw_N_bitwise": bool(exact_n),
            "sample_W_exact": exact_w, "full_output_scans": full}


def launch_case(run_dir, gpu_uuid, exchange_root, cache_root, dcf_path,
                manifest, day, method, category, block=None, label=None):
    name = label or f"day{day}_{category}{'' if block is None else f'_p{block}'}_{method}"
    final = run_dir / "cases" / name
    if final.exists():
        for record in read_json(final / "manifest.json"):
            verify(final / record["path"], record)
        result = read_json(final / "result.json")
        if (result["day"], result["method"], result["category"], result["block"]) != (day, method, category, block) or not result["gate"]["passed"]:
            raise RuntimeError(f"committed case identity or gate mismatch: {name}")
        return result
    if any((run_dir / "attempts").glob(name + ".*")):
        raise RuntimeError(f"previous attempt for {name} was preserved; inspect it before retrying")
    boundary(run_dir, gpu_uuid, "before_" + name)
    folder = Path(tempfile.mkdtemp(prefix=name + ".", dir=run_dir / "attempts"))
    binary = TASK / "build" / BINARIES[method]
    command = [str(binary), f"--gpus={gpu_uuid}", f"--cache_root={cache_root}",
               f"--out_dir={folder}", "--nside=4096", f"--day={day}", "--max_segments=10"]
    refs = manifest["references"][str(day)]
    if method == "B":
        command += [f"--sample_dir={(exchange_root / refs['ids']['path']).parent}", f"--dcf_bin={dcf_path}"]
    else:
        command += ["--rep_phase=grid"]
    write_json(folder / "command.json", {"argv": command, "gpu_uuid": gpu_uuid, "at": utc_now(),
                                         "day": day, "method": method, "category": category, "block": block,
                                         "binary_sha256": digest(binary)})
    write_json(run_dir / "status.json", {"state": "running", "case": name, "at": utc_now()})
    print(f"{utc_now()} start {name}", flush=True)
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "8",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (folder / "run.log").open("wb") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=1800)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
    marker = "FULL_INVERSE_DONE" if method == "B" else "FULL_STS_DONE"
    if rc != 0 or (folder / "run.log").read_text().count(marker) != 1:
        raise RuntimeError(f"{name} failed with rc={rc}; see {folder / 'run.log'}")
    metrics = read_json(folder / "metrics.json")
    if (metrics["method"], metrics["day"], metrics["nside"], metrics["segments"],
            metrics.get("output_pixels", metrics.get("pixels"))) != (METHOD_IDS[method], day, 4096, 10, PIXELS):
        raise RuntimeError(f"{name} output metrics have an unexpected scope")
    if not math.isfinite(metrics["inverse_loop_wall_s"]) or metrics["inverse_loop_wall_s"] <= 0:
        raise RuntimeError(f"{name} produced an invalid inverse-loop time")
    with (folder / "segments.csv").open() as handle:
        if [int(row["segment"]) for row in csv.DictReader(handle)] != list(range(1, 11)):
            raise RuntimeError(f"{name} is missing original segments")
    gate = scientific_gate(folder, day, method, refs, exchange_root)
    result = {"name": name, "day": day, "method": method, "category": category, "block": block,
              "at": utc_now(), "metrics": metrics, "gate": gate}
    write_json(folder / "result.json", result)
    if not gate["passed"]:
        raise RuntimeError(f"{name} failed its numerical gate; diagnostics preserved in {folder}")
    write_json(folder / "manifest.json", records_under(folder))
    folder.replace(final)
    boundary(run_dir, gpu_uuid, "after_" + name)
    print(f"{utc_now()} {name}: {metrics['inverse_loop_wall_s']:.6f}s", flush=True)
    return result


def summarize(run_dir, records):
    columns = ("day", "block", "method", "inverse_loop_wall_s", "output_finalize_wall_s")
    with (run_dir / "formal_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({**{key: row[key] for key in columns[:3]},
                             **{key: row["metrics"][key] for key in columns[3:]}})
    days = {}
    for day in DAYS:
        grouped = {m: {r["block"]: r["metrics"]["inverse_loop_wall_s"] for r in records
                       if r["day"] == day and r["method"] == m} for m in BINARIES}
        b, c = grouped["B"], grouped["C"]
        paired = [b[block] / c[block] for block in sorted(b.keys() & c.keys())]
        days[str(day)] = {"B_raw_s": list(b.values()), "C_raw_s": list(c.values()),
                          "B_median_s": statistics.median(b.values()) if b else None,
                          "C_median_s": statistics.median(c.values()) if c else None,
                          "B_over_C_paired_cost_ratios": paired,
                          "B_over_C_paired_median": statistics.median(paired) if paired else None,
                          "B_over_C_method_median_ratio": statistics.median(b.values()) / statistics.median(c.values()) if b and c else None}
    write_json(run_dir / "summary.json",
               {"state": "completed" if len(records) == 40 else "partial", "n_formal": len(records), "days": days})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-root", required=True, type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--dcf-bin", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--phase", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--run-dir", type=Path, default=TASK / "run")
    parser.add_argument("--verify-headers", action="store_true")
    args = parser.parse_args()
    manifest = read_json(TASK / "input_identities.json")
    if tuple(manifest["days"]) != DAYS:
        raise RuntimeError("run-day manifest mismatch")
    exchange_root = args.exchange_root.resolve()
    for record in manifest["header_files"]:
        verify(exchange_root / record["path"], record)
    if args.verify_headers:
        print("Frozen header identities verified", flush=True)
        return
    if not args.gpu_uuid or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        parser.error("a full GPU UUID is required")
    cache_root = (args.cache_root or (exchange_root / Path(manifest["caches"][0]["path"]).parent)).resolve()
    dcf_path = (args.dcf_bin or Path(manifest["dcf"]["path"])).resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.parent != TASK or run_dir.name in {"build", "source"}:
        parser.error("the new run directory must be a direct child of the follow-up task directory")
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in ("attempts", "cases", "boundaries"):
            (run_dir / folder).mkdir(exist_ok=True)
        try:
            boundary(run_dir, args.gpu_uuid, "preflight")
            verify_inputs(exchange_root, cache_root, dcf_path, manifest)
            frozen = {"exchange_root": str(exchange_root), "cache_root": str(cache_root),
                      "dcf_path": str(dcf_path), "gpu_uuid": args.gpu_uuid,
                      "days": list(DAYS), "paired_order": [list(pair) for pair in PAIRS],
                      "gates_per_day_method": 1, "warmups_per_day_method": 1,
                      "files": {str(path.relative_to(TASK)): digest(path) for path in
                                [TASK / "build" / name for name in BINARIES.values()] +
                                [TASK / name for name in ("build.sh", "input_identities.json", "run_five_day_timing.py",
                                                          "replay_3d_full.cu", "replay_sts_full.cu", "PROTOCOL.md")]}}
            freeze = run_dir / "freeze.json"
            if freeze.exists():
                if read_json(freeze) != frozen:
                    raise RuntimeError("input, configuration, source or binary changed after freeze")
            elif args.phase == "run":
                raise RuntimeError("run requires a successful preflight in this run directory")
            else:
                write_json(freeze, frozen)
            if args.phase == "preflight":
                write_json(run_dir / "status.json", {"state": "preflight_verified", "at": utc_now()})
                print("Preflight verified all 50 caches, references, device and build identities", flush=True)
                return
            records = []
            for day in DAYS:
                launch = (run_dir, args.gpu_uuid, exchange_root, cache_root, dcf_path, manifest, day)
                for category in ("gate", "warmup"):
                    for method in BINARIES:
                        launch_case(*launch, method, category)
                expected = [run_dir / "cases" / f"day{day}_formal_p{block}_{method}"
                            for block, pair in enumerate(PAIRS, 1) for method in pair]
                if any(path.exists() for path in expected) and not all(path.exists() for path in expected):
                    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
                    for method in BINARIES:
                        launch_case(*launch, method, "resume_warmup",
                                    label=f"day{day}_resume_warmup_{stamp}_{method}")
                for block, pair in enumerate(PAIRS, 1):
                    for method in pair:
                        records.append(launch_case(*launch, method, "formal", block))
                        summarize(run_dir, records)
            write_json(run_dir / "status.json", {"state": "completed", "n_formal": 40, "at": utc_now()})
        except BaseException as error:
            write_json(run_dir / "status.json", {"state": "paused_or_failed", "at": utc_now(), "reason": str(error)})
            raise


if __name__ == "__main__":
    main()
