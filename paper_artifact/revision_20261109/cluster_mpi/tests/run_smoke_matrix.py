#!/usr/bin/env python3
"""Run bounded same-host MPI/CUDA integration checks (not a benchmark).

Only the process groups created by this script are stopped on timeout. There
is no GPU polling or remote monitoring. A one-time resource preflight belongs
to the external launcher; all GPU ranks here share the explicitly selected GPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

from compare_runs import compare, sha256, VerificationError


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute(command, logdir, env, timeout):
    logdir.mkdir(parents=True, exist_ok=False)
    started = utc_now()
    t0 = time.monotonic()
    with (logdir / "stdout.log").open("wb") as stdout, (logdir / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait()
    record = {"command": command, "started_at": started, "finished_at": utc_now(),
              "exit_code": code, "timed_out": timed_out, "elapsed_progress_seconds": time.monotonic() - t0,
              "pid": child.pid, "process_group": child.pid,
              "logdir": str(logdir), "performance_claim": False}
    atomic_json(logdir / "execution.json", record)
    return record


def make_sky(nside, root):
    directory = root / f"nside{nside}"
    directory.mkdir(parents=True, exist_ok=False)
    # Deterministic nonuniform positive input plus three unequal bright pixels.
    # NumPy owns only the input values; production CUDA owns the NEST geometry.
    i = np.arange(12 * nside * nside, dtype=np.uint64)
    values = (0.1 + ((i * 37 + 11) % 1009).astype(np.float64) / 1009.0).astype("<f4")
    values[0], values[len(values) // 3], values[-1] = 9.0, 4.0, 2.0
    path = directory / "B_10M.bin"
    values.tofile(path)
    return {"nside": nside, "npix": len(values), "path": str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": sha256(path),
            "dtype": "little_endian_float32", "ordering": "values indexed by production CUDA NEST pixel ID",
            "rule": "float32(0.1+((uint64(pixel_id)*37+11)%1009)/1009); pixel0=9,pixelN/3=4,pixelN-1=2"}


def verify_same_host_topology(output, strategy, ranks):
    metrics = json.loads((output / "metrics.json").read_text())
    if metrics.get("host_count") != 1 or metrics.get("cross_node_execution", False):
        raise VerificationError("same-host smoke unexpectedly reports real cross-node execution")
    rankinfo = metrics.get("rank_info", [])
    if len(rankinfo) != ranks or len({x["gpu_uuid"] for x in rankinfo}) != 1:
        raise VerificationError("same-GPU smoke must report all ranks on exactly one GPU UUID")
    if metrics.get("config", {}).get("strategy", "flat") != strategy:
        raise VerificationError("driver strategy differs from frozen smoke command")
    topology = metrics.get("node_topology")
    if strategy == "hierarchical":
        expected = [list(range(i, min(i + 2, ranks))) for i in range(0, ranks, 2)]
        expected_leaders = [2 * (i // 2) for i in range(ranks)]
        if (not topology or topology.get("simulated") is not True
                or topology.get("validation_node_size") != 2
                or topology.get("groups") != expected
                or topology.get("leader_world_rank_by_rank") != expected_leaders
                or topology.get("node_group_count") != len(expected)
                or topology.get("actual_shared_memory_group_count") != 1
                or metrics.get("actual_host_count") != 1):
            raise VerificationError("hierarchical logical-group topology does not match bounded validation plan")
    return {"actual_host_count": metrics.get("actual_host_count", metrics["host_count"]),
            "physical_gpu_uuid_count": 1, "gpu_name": rankinfo[0]["gpu_name"],
            "strategy": strategy, "node_topology": topology, "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--original-binary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, help="physical index for this explicitly selected single GPU")
    parser.add_argument("--mpi-launcher", default="mpiexec")
    parser.add_argument("--mpi-extra", default="", help="launcher-specific words, e.g. --oversubscribe for OpenMPI only")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--skip-negative", action="store_true")
    parser.add_argument("--strategies", default="flat",
                        help="Distinct comma-separated flat,hierarchical; hierarchical uses tiny logical groups of2 on one GPU")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output-dir already exists; use a new attempt, never overwrite earlier results")
    if not args.binary.is_file() or not args.original_binary.is_file():
        parser.error("both executable binaries must exist")
    if args.gpu < 0 or args.timeout_seconds <= 0:
        parser.error("gpu must be nonnegative and timeout positive")
    strategies = args.strategies.split(",")
    if (not strategies or len(set(strategies)) != len(strategies)
            or any(x not in ("flat", "hierarchical") for x in strategies)):
        parser.error("strategies must be distinct flat and/or hierarchical")
    root = args.output_dir.resolve()
    root.mkdir(parents=True)
    args.binary, args.original_binary = args.binary.resolve(), args.original_binary.resolve()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    launcher = [args.mpi_launcher, *shlex.split(args.mpi_extra)]
    state = {"schema": "cluster_same_host_smoke_matrix_v1", "status": "running",
             "started_at": utc_now(), "same_host": True, "physical_gpu_indices": [args.gpu],
             "cross_node_test": False, "a4000_test": False, "performance_claim": False,
             "continuous_resource_monitoring": False,
             "source_sha256": {"run_smoke_matrix.py": sha256(__file__),
                               "compare_runs.py": sha256(Path(__file__).with_name("compare_runs.py"))},
             "binary_sha256": {"cluster": sha256(args.binary), "original": sha256(args.original_binary)},
             "strategies": strategies, "hierarchical_validation_node_size": 2 if "hierarchical" in strategies else None,
             "fixtures": [], "positive_runs": [], "comparisons": [], "cross_strategy_comparisons": [], "negative_runs": []}
    statepath = root / "smoke_matrix.json"
    atomic_json(statepath, state)
    try:
        fixtures_root = root / "fixtures"
        fixtures_root.mkdir()
        for nside in (8, 16):
            sky = make_sky(nside, fixtures_root)
            state["fixtures"].append(sky)
            atomic_json(statepath, state)
            for mode, occ_mode in (("rep", 2), ("exact", 0)):
                strategy_paths = {}
                rank_counts = (1, 2, 3, 6) if nside == 16 and "hierarchical" in strategies else (1, 2, 3)
                for strategy in strategies:
                    case = f"nside{nside}_{mode}"
                    if len(strategies) != 1 or strategy != "flat":
                        case += "_" + strategy
                    paths = []
                    strategy_paths[strategy] = {}
                    for ranks in rank_counts:
                        output = root / "runs" / f"{case}_np{ranks}"
                        command = [*launcher, "-n", str(ranks), str(args.binary),
                                   f"--nside={nside}", f"--input={sky['path']}",
                                   "--day_start=1", "--day_count=1", "--segs=2",
                                   "--time_start=0", "--time_count=8", "--orbit_seed=42",
                                   "--grid_cell_scale=1.0", f"--recon-mode={mode}",
                                   "--allow-shared-gpu=1", "--serial-check=1", "--dump-validation=1"]
                        if len(strategies) != 1 or strategy != "flat":
                            command.append("--strategy=" + strategy)
                        if strategy == "hierarchical":
                            command.append("--validation-node-size=2")
                        command.append(f"--output-dir={output}")
                        result = execute(command, root / "logs" / f"{case}_np{ranks}", env, args.timeout_seconds)
                        state["positive_runs"].append({"case": case, "strategy": strategy,
                                                       "ranks": ranks, "output": str(output),
                                                       "logical_groups_simulated": ([min(2, ranks-i) for i in range(0, ranks, 2)]
                                                                                    if strategy == "hierarchical" else None),
                                                       "input_sha256": sky["sha256"], **result})
                        atomic_json(statepath, state)
                        if result["timed_out"] or result["exit_code"] != 0:
                            raise VerificationError(f"MPI positive run failed: {case} ranks={ranks}")
                        state["positive_runs"][-1]["execution_topology"] = verify_same_host_topology(output, strategy, ranks)
                        atomic_json(statepath, state)
                        paths.append(output)
                        strategy_paths[strategy][ranks] = output
                    reference_dir = root / "original" / case
                    command = [str(args.original_binary), "--btag=10M", f"--nside={nside}",
                               "--day_start=1", "--day_count=1", "--segs=2",
                               "--time_start=0", "--time_count=8", "--orbit_seed=42",
                               "--grid_cell_scale=1.0", "--viss_tile_pix=256", "--blockage=1",
                               f"--occ_mode={occ_mode}", f"--gpus={args.gpu}",
                               f"--sky_dir={Path(sky['path']).parent}", f"--out_dir={reference_dir}",
                               "--B_mode=bin", "--C_mode=bin", "--write_output=1",
                               "--write_viss=0", "--profile=0", "--print_stats=0"]
                    result = execute(command, root / "logs" / f"{case}_original", env, args.timeout_seconds)
                    state["positive_runs"].append({"case": case, "strategy": strategy,
                                                   "kind": "immutable_original_pipeline",
                                                   "input_sha256": sky["sha256"], **result})
                    atomic_json(statepath, state)
                    if result["timed_out"] or result["exit_code"] != 0:
                        raise VerificationError(f"original reference failed: {case}")
                    comparison = compare(paths, reference_dir / "C1day10M.bin")
                    comparison["input_sha256"] = sky["sha256"]
                    comparison_path = root / "comparisons" / f"{case}.json"
                    comparison_path.parent.mkdir(exist_ok=True)
                    atomic_json(comparison_path, comparison)
                    state["comparisons"].append({"case": case, "strategy": strategy, "path": str(comparison_path),
                                                 "sha256": sha256(comparison_path), "status": "passed"})
                    atomic_json(statepath, state)
                if "flat" in strategy_paths and "hierarchical" in strategy_paths:
                    for ranks in rank_counts:
                        comparison = compare([strategy_paths["flat"][ranks], strategy_paths["hierarchical"][ranks]],
                                             allow_strategy_difference=True)
                        comparison["input_sha256"] = sky["sha256"]
                        case = f"nside{nside}_{mode}_np{ranks}_flat_vs_hierarchical"
                        comparison_path = root / "comparisons" / f"{case}.json"
                        atomic_json(comparison_path, comparison)
                        state["cross_strategy_comparisons"].append({"case": case, "path": str(comparison_path),
                                                                    "sha256": sha256(comparison_path), "status": "passed"})
                        atomic_json(statepath, state)

        if not args.skip_negative:
            negatives = [
                ("nside_not_power_of_two", 1, ["--nside=3", "--synthetic=1", "--allow-shared-gpu=1"]),
                ("unknown_reconstruction_mode", 1, ["--nside=8", "--synthetic=1", "--recon-mode=invalid", "--allow-shared-gpu=1"]),
                ("missing_input", 1, ["--nside=8", f"--input={root / 'missing_input.bin'}", "--allow-shared-gpu=1"]),
                ("negative_time_count", 1, ["--nside=8", "--synthetic=1", "--time_count=-1", "--allow-shared-gpu=1"]),
                ("more_ranks_than_native_tiles", 4, ["--nside=8", "--synthetic=1", "--allow-shared-gpu=1"]),
                ("shared_gpu_requires_explicit_consent", 2, ["--nside=8", "--synthetic=1"]),
            ]
            for name, ranks, options in negatives:
                output = root / "negative_outputs" / name
                time_options = [] if any(x.startswith("--time_count=") for x in options) else ["--time_count=2"]
                command = [*launcher, "-n", str(ranks), str(args.binary), *options,
                           "--segs=1", *time_options, f"--output-dir={output}"]
                result = execute(command, root / "logs" / f"negative_{name}", env, min(args.timeout_seconds, 45))
                # A hung collective is a failure, not successful rejection.
                passed = result["exit_code"] != 0 and not result["timed_out"]
                state["negative_runs"].append({"case": name, "passed": passed, **result})
                atomic_json(statepath, state)
                if not passed:
                    raise VerificationError(f"invalid configuration not promptly rejected: {name}")

        state["status"] = "passed"
        state["qualification"] = "MPI ranks ran on one physical4090 only; real network and A4000 deployment remain untested."
    except (VerificationError, OSError, ValueError, KeyError, TypeError) as exc:
        state["status"] = "failed"
        state["failure"] = str(exc)
    finally:
        state["finished_at"] = utc_now()
        atomic_json(statepath, state)
    print(json.dumps({"status": state["status"], "report": str(statepath),
                      "failure": state.get("failure")}, ensure_ascii=False))
    return 0 if state["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
