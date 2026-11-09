#!/usr/bin/env python3
"""Repeat three fixed grid/segment cases on the complete day-1 STS output domain."""
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
CASES = ("s10_g1", "s20_g1", "s10_g1p5")
FORMAL_ORDER = tuple(CASES[i:]+CASES[:i] for i in range(3))
BINARY = TASK / "build/replay_sts_sweep"
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


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def frozen_cache_plans(cache_root, manifest):
    payloads = []
    for segment, record in enumerate(manifest["caches"], 1):
        path = cache_root / Path(record["path"]).name
        contents = path.read_bytes()
        if len(contents) != record["bytes"] or hashlib.sha256(contents).hexdigest() != record["sha256"]:
            raise RuntimeError(f"frozen cache identity mismatch: {path}")
        magic, *header = struct.unpack("<8s7i", contents[:36])
        if magic != b"PREPLAY1" or header != [4096, 1, segment, (segment-1)*6981, 6981, 195468, 1]:
            raise RuntimeError(f"wrong cache header: {path}")
        payloads.append(contents[36:])
    rows = b"".join(payloads)
    if len(rows) != 1954680 * 60:
        raise RuntimeError("cache does not contain every frozen day-1 record")
    plans = {}
    for count in (10, 20):
        t0 = 0
        segments = []
        for s in range(count):
            tlen = 69810 // count + (s < 69810 % count)
            raw = rows[t0*28*60:(t0+tlen)*28*60]
            segments.append({"segment": str(s+1), "t0": str(t0), "tlen": str(tlen),
                             "half": str(28*tlen), "payload_sha256": hashlib.sha256(raw).hexdigest()})
            t0 += tlen
        plans[str(count)] = segments
    return {"day_payload_sha256": hashlib.sha256(rows).hexdigest(), "segment_plans": plans}


def verify_result(folder, setting, plan, exchange_root):
    import numpy as np

    metrics = read_json(folder / "metrics.json")
    scope = (metrics["method"], metrics["day"], metrics["nside"], metrics["pixels"],
             metrics["half_baselines"], metrics["segments"], metrics["grid_scale"])
    if scope != ("sts_grid", 1, 4096, PIXELS, 1954680, setting["segments"], setting["grid_scale"]):
        raise RuntimeError(f"unexpected output scope: {scope}")
    if not math.isfinite(metrics["inverse_loop_wall_s"]) or metrics["inverse_loop_wall_s"] <= 0:
        raise RuntimeError("invalid inverse-loop time")
    if read_csv(folder / "cache_plan.csv") != plan:
        raise RuntimeError("resegmented input payload SHA or partition mismatch")
    actual_groups = read_csv(folder / "segments.csv")
    reference_groups = read_csv(exchange_root / setting["groups"]["path"])
    if len(actual_groups) != len(reference_groups) or len(actual_groups) != setting["segments"]:
        raise RuntimeError("group metadata row count differs")
    for actual, reference in zip(actual_groups, reference_groups):
        if actual["segment"] != reference["seg"] or any(
                actual[name] != reference[name] for name in ("t0", "tlen", "half", "groups", "RES")):
            raise RuntimeError(f"partition or occupied-cell identity mismatch at segment {actual['segment']}")
        if any(not math.isclose(float(actual[name]), float(reference[name]), rel_tol=2e-11, abs_tol=1e-12)
               for name in ("fa", "fb", "du")):
            raise RuntimeError(f"plane fit or grid spacing differs at segment {actual['segment']}")
    ids = np.fromfile(folder / "sample_ids.u32", dtype="<u4")
    numerator = np.fromfile(folder / "sample_N.f32", dtype="<f4")
    weights = np.fromfile(folder / "sample_W.u32", dtype="<u4")
    with np.load(exchange_root / setting["samples"]["path"], allow_pickle=False) as reference:
        expected = reference["rep_N"].astype(np.float64)
        if ids.shape != (4096,) or numerator.shape != ids.shape or not np.array_equal(ids, reference["pixel_ids"]):
            raise RuntimeError("fixed 4096-sample identities differ")
        if not np.array_equal(weights, reference["rep_W"]):
            raise RuntimeError("sample W differs from archived representative STS")
    error = numerator.astype(np.float64) - expected
    l2 = float(np.linalg.norm(error) / np.linalg.norm(expected))
    peak = float(np.max(abs(error)) / np.max(abs(expected)))
    if not (l2 <= 1e-6 and peak <= 1e-6):
        raise RuntimeError(f"sample N exceeds archived replay tolerance: L2={l2}, max/peak={peak}")
    full = read_json(folder / "full_stats.json")
    if any(full[key]["count"] != PIXELS or full[key]["nonfinite"] != 0 or
           not re.fullmatch(r"[0-9a-f]{64}", full[key]["sha256"]) for key in ("rawN", "W", "C")):
        raise RuntimeError("full-output scan failed")
    return {"metrics": metrics, "sample_raw_N_relative_l2": l2, "sample_raw_N_max_over_peak": peak,
            "sample_W_exact": True, "total_groups": sum(int(row["groups"]) for row in actual_groups),
            "full_hashes": {key: full[key]["sha256"] for key in ("rawN", "W", "C")}}


def launch(run_dir, gpu_uuid, exchange_root, cache_root, plans, settings,
           case, category, block=None, warm_hashes=None, suffix=""):
    name = f"{case}_{category}{'' if block is None else f'_b{block}'}{suffix}"
    final = run_dir / "cases" / name
    identity = (case, category, block)
    if final.exists():
        for record in read_json(final / "manifest.json"):
            verify(final / record["path"], record)
        result = read_json(final / "result.json")
        if (result["case"], result["category"], result["block"]) != identity:
            raise RuntimeError(f"committed case identity differs: {name}")
        if warm_hashes is not None and result["full_hashes"] != warm_hashes:
            raise RuntimeError(f"committed case output differs from warm-up: {name}")
        return result
    if any((run_dir / "attempts").glob(name + ".*")):
        raise RuntimeError(f"retained unfinished attempt for {name}; inspect before retrying")
    boundary(run_dir, gpu_uuid, "before_" + name)
    attempt = Path(tempfile.mkdtemp(prefix=name + ".", dir=run_dir / "attempts"))
    setting = settings[case]
    command = [str(BINARY), f"--gpus={gpu_uuid}", f"--cache_root={cache_root}",
               f"--out_dir={attempt}", f"--segs={setting['segments']}", f"--grid_scale={setting['grid_scale']}"]
    write_json(attempt / "command.json", {"argv": command, "identity": identity, "at": utc_now(),
                                          "binary_sha256": digest(BINARY)})
    write_json(run_dir / "status.json", {"state": "running", "case": name, "at": utc_now()})
    print(f"{utc_now()} start {name}", flush=True)
    env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "OMP_NUM_THREADS": "8",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with (attempt / "run.log").open("wb") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = proc.wait(timeout=1200)
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            raise
    if rc or (attempt / "run.log").read_text().count("FULL_STS_DONE") != 1:
        raise RuntimeError(f"{name} failed, rc={rc}; log at {attempt / 'run.log'}")
    result = verify_result(attempt, setting, plans[str(setting["segments"])], exchange_root)
    result.update({"case": case, "category": category, "block": block, "at": utc_now()})
    write_json(attempt / "result.json", result)
    if warm_hashes is not None and result["full_hashes"] != warm_hashes:
        raise RuntimeError(f"{name} full N/W/C hashes differ from its accepted warm-up")
    write_json(attempt / "manifest.json", records_under(attempt))
    attempt.replace(final)
    boundary(run_dir, gpu_uuid, "after_" + name)
    print(f"{utc_now()} {name}: {result['metrics']['inverse_loop_wall_s']:.6f}s", flush=True)
    return result


def summarize(run_dir, records):
    columns = ("case", "block", "total_groups", "inverse_loop_wall_s", "output_finalize_wall_s")
    csv_path = run_dir / "formal_runs.csv"
    temporary = csv_path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({**{key: row[key] for key in columns[:3]},
                             **{key: row["metrics"][key] for key in columns[3:]}})
    temporary.replace(csv_path)
    stats = {}
    for case in CASES:
        values = [row["metrics"]["inverse_loop_wall_s"] for row in records if row["case"] == case]
        if values:
            stats[case] = {"raw_inverse_s": values, "median_s": statistics.median(values),
                           "sample_sd_s": statistics.stdev(values) if len(values) > 1 else None,
                           "min_s": min(values), "max_s": max(values)}
    write_json(run_dir / "summary.json", {"state": "completed" if len(records) == 9 else "partial",
                                          "n_formal": len(records), "cases": stats})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exchange-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--phase", choices=("preflight", "run"), default="preflight")
    parser.add_argument("--verify-small-inputs", action="store_true")
    args = parser.parse_args()
    root = args.exchange_root.resolve()
    manifest = read_json(TASK / "input_identities.json")
    settings = manifest["cases"]
    if tuple(settings) != CASES or len(manifest["caches"]) != 10:
        raise RuntimeError("frozen sweep matrix mismatch")
    for record in manifest["header_files"] + [manifest["full_wrapper_origin"], manifest["sampled_replay_origin"]]:
        verify(root / record["path"], record)
    for setting in settings.values():
        for key in ("samples", "groups"):
            record = setting[key]
            verify(root / record["path"], record)
    if args.verify_small_inputs:
        print("Frozen source and sample/group identities verified", flush=True)
        return
    if not args.gpu_uuid or not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", args.gpu_uuid):
        parser.error("a full GPU UUID is required")
    import numpy  # Verify the scientific-gate dependency before GPU work.
    cache = (args.cache_root or (root / Path(manifest["caches"][0]["path"]).parent)).resolve()
    run_dir = TASK / "run"
    if run_dir.is_symlink():
        raise RuntimeError("run directory must reside inside this task")
    run_dir.mkdir(exist_ok=True)
    with (run_dir / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for folder in ("attempts", "cases", "boundaries"):
            (run_dir / folder).mkdir(exist_ok=True)
        try:
            boundary(run_dir, args.gpu_uuid, "preflight")
            payload = frozen_cache_plans(cache, manifest)
            frozen = {"gpu_uuid": args.gpu_uuid, "exchange_root": str(root), "cache_root": str(cache),
                      "formal_order": [list(block) for block in FORMAL_ORDER], "payload": payload,
                      "file_sha256": {name: digest(TASK / name) for name in
                                     ("input_identities.json", "build.sh", "PROTOCOL.md", "replay_sts_sweep.cu",
                                      "run_sweep_repeats.py", "build/replay_sts_sweep")}}
            freeze = run_dir / "freeze.json"
            if freeze.exists() and read_json(freeze) != frozen:
                raise RuntimeError("frozen input, GPU, protocol, source or binary changed")
            if args.phase == "preflight":
                write_json(freeze, frozen)
                write_json(run_dir / "status.json", {"state": "preflight_verified", "at": utc_now()})
                print("Frozen full-day payload, segmentation plans, device and build verified", flush=True)
                return
            if not freeze.is_file():
                raise RuntimeError("run requires a completed preflight")
            base = (run_dir, args.gpu_uuid, root, cache, payload["segment_plans"], settings)
            warms = {case: launch(*base, case, "warmup")["full_hashes"] for case in CASES}
            formal = [run_dir / "cases" / f"{case}_formal_b{block}" for block, order in enumerate(FORMAL_ORDER, 1) for case in order]
            if any(path.exists() for path in formal) and not all(path.exists() for path in formal):
                suffix = "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
                for case in CASES:
                    launch(*base, case, "resume_warmup", warm_hashes=warms[case], suffix=suffix)
            records = []
            for block, order in enumerate(FORMAL_ORDER, 1):
                for case in order:
                    records.append(launch(*base, case, "formal", block, warms[case]))
                    summarize(run_dir, records)
            write_json(run_dir / "status.json", {"state": "completed", "n_formal": 9, "at": utc_now()})
        except BaseException as error:
            write_json(run_dir / "status.json", {"state": "paused_or_failed", "at": utc_now(), "reason": str(error)})
            raise


if __name__ == "__main__":
    main()
