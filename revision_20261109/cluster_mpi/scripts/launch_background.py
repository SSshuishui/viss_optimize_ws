#!/usr/bin/env python3
"""Submit a bounded serial MPI queue, without GPU/progress polling."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time

from build import ROOT, environment, sha


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stop_own_group(process):
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def worker(config_path):
    config = json.loads(config_path.read_text())
    if config.get("prepare_only"):
        raise RuntimeError("a prepare-only plan cannot be executed as a worker; submit a fresh run without --prepare-only")
    run_dir = Path(config["run_dir"])
    status_file = run_dir / "status.json"
    state = {"schema": "cluster_mpi_queue_status_v1", "state": "running", "pid": os.getpid(),
             "started_at_utc": now(), "continuous_resource_monitoring": False, "steps": []}
    active = None

    def interrupted(signum, _frame):
        if active is not None:
            stop_own_group(active)
        raise KeyboardInterrupt("signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    write_json(status_file, state)
    env = environment()
    env.update(config["environment"])
    try:
        for step in config["steps"]:
            directory = run_dir / step["name"]
            directory.mkdir()
            record = {"name": step["name"], "state": "running", "started_at_utc": now(),
                      "command": step["command"], "stdout_stderr": str(directory / "run.log")}
            state["steps"].append(record)
            write_json(status_file, state)
            with (directory / "run.log").open("w") as log:
                log.write(shlex.join(step["command"]) + "\n")
                log.flush()
                active = subprocess.Popen(step["command"], env=env, cwd=ROOT, stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = active.pid
                write_json(status_file, state)
                try:
                    return_code = active.wait(timeout=config["timeout_s"])
                except subprocess.TimeoutExpired:
                    stop_own_group(active)
                    raise RuntimeError("step timed out: " + step["name"])
                finally:
                    record["exit_code"] = active.poll()
                    active = None
            record["finished_at_utc"] = now()
            if return_code:
                raise RuntimeError("MPI step failed with exit code " + str(return_code))
            metrics_file = directory / "output/metrics.json"
            metrics = json.loads(metrics_file.read_text())
            if metrics.get("status") not in (None, "completed", "ok", "passed"):
                raise RuntimeError("driver metrics status is not successful: " + str(metrics.get("status")))
            record.update(state="completed", metrics=str(metrics_file), metrics_sha256=sha(metrics_file))
            write_json(status_file, state)
        state["state"] = "completed"
        state["exit_code"] = 0
    except BaseException as error:
        if active is not None:
            stop_own_group(active)
        if state["steps"] and state["steps"][-1]["state"] == "running":
            state["steps"][-1]["state"] = "failed"
        state.update(state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     exit_code=130 if isinstance(error, KeyboardInterrupt) else 1, failure_reason=str(error))
    finally:
        state["finished_at_utc"] = now()
        write_json(status_file, state)
        try:
            (run_dir / ".queue.lock").rmdir()
        except OSError:
            pass
    return state["exit_code"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--mode", choices=("local-smoke", "cluster"), default="local-smoke")
    p.add_argument("--ranks", help="Explicit total MPI ranks, comma separated; mutually exclusive with --nodes")
    p.add_argument("--nodes", help="Cluster node counts, comma separated; ranks = nodes * gpus-per-node")
    p.add_argument("--gpus-per-node", type=int, choices=(1, 2), default=1)
    p.add_argument("--recon-modes", default="rep,exact")
    p.add_argument("--strategies", default="flat", help="flat,hierarchical; each selected strategy is a separate step")
    p.add_argument("--physical-gpu", help="An explicitly selected idle/shared GPU index or UUID for local smoke")
    p.add_argument("--min-free-mib", type=int, default=2048)
    p.add_argument("--hostfile", type=Path)
    p.add_argument("--binary", type=Path, default=ROOT / "build/cluster_sts_mpi")
    p.add_argument("--mpi-launcher", default=os.environ.get("MPIEXEC"))
    p.add_argument("--mpi-flavor", choices=("auto", "mpich", "openmpi"), default="auto")
    p.add_argument("--mpi-arg", action="append", default=[], help="Additional exact launcher argument; use --mpi-arg=VALUE")
    p.add_argument("--driver-arg", action="append", default=[], help="Override/add driver argument, e.g. --driver-arg=--nside=64")
    p.add_argument("--timeout-s", type=int, default=600)
    p.add_argument("--foreground", action="store_true", help="Short smoke/debug only; no active resource monitoring")
    p.add_argument("--prepare-only", action="store_true", help="Write a CPU-only plan; no MPI/SMI invocation, lock, worker or GPU launch")
    a = p.parse_args()
    if a.worker:
        return worker(a.worker.resolve())
    if a.run_dir is None:
        p.error("--run-dir is required and must not exist")
    if a.timeout_s < 1 or a.min_free_mib < 0:
        p.error("timeout must be positive and memory budget nonnegative")
    if a.ranks is not None and a.nodes is not None:
        p.error("--nodes and --ranks are mutually exclusive")
    if a.mode == "local-smoke" and (a.nodes is not None or a.gpus_per_node != 1):
        p.error("local-smoke is a shared single-GPU check; use cluster mode for --nodes or two GPUs per node")
    if a.prepare_only and a.mpi_flavor == "auto":
        p.error("--prepare-only requires explicit --mpi-flavor=mpich|openmpi so no MPI probe is invoked")
    if a.prepare_only and a.foreground:
        p.error("--prepare-only cannot be combined with --foreground")
    counts = a.ranks if a.ranks is not None else (a.nodes if a.nodes is not None else "1,2,3")
    try:
        values = [int(x) for x in counts.split(",")]
    except ValueError:
        p.error("--nodes/--ranks must contain positive integers separated by commas")
    if not values or any(x < 1 for x in values) or len(set(values)) != len(values):
        p.error("--nodes/--ranks must contain distinct positive integers")
    if a.mode == "cluster" and a.ranks is None:
        ranks = [n * a.gpus_per_node for n in values]
    else:
        ranks = values
    if a.mode == "cluster" and any(n % a.gpus_per_node for n in ranks):
        p.error("each explicit cluster rank count must be divisible by --gpus-per-node")
    modes = a.recon_modes.split(",")
    if any(x not in ("rep", "exact") for x in modes):
        p.error("recon modes are rep/exact")
    strategies = a.strategies.split(",")
    if any(x not in ("flat", "hierarchical") for x in strategies) or len(set(strategies)) != len(strategies):
        p.error("strategies must be distinct flat/hierarchical values")
    if a.mode == "local-smoke" and a.physical_gpu is None:
        p.error("local smoke requires explicit --physical-gpu; no GPU is selected automatically")
    if a.mode == "cluster" and (a.hostfile is None or not a.hostfile.is_file()):
        p.error("cluster mode requires an actual --hostfile; do not use the unchanged example")
    if a.mode == "cluster" and a.hostfile.name.endswith(".example") and not a.prepare_only:
        p.error("use an actual hostfile, not the supplied example")
    binary = a.binary.resolve()
    if not binary.is_file() and not a.prepare_only:
        p.error("binary missing; build first: " + str(binary))
    env = environment()
    launcher = a.mpi_launcher or shutil.which("mpiexec", path=env["PATH"]) or shutil.which("mpirun", path=env["PATH"])
    if not launcher and a.prepare_only:
        launcher = "mpirun" if a.mpi_flavor == "openmpi" else "mpiexec"
    if not launcher:
        p.error("MPI launcher missing; set MPIEXEC or install an isolated dependency")
    version = None
    if not a.prepare_only:
        version = subprocess.run([launcher, "--version"], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
    openmpi = a.mpi_flavor == "openmpi" or (a.mpi_flavor == "auto" and bool(re.search(r"Open MPI|OpenRTE", version or "", re.I)))
    overrides = {}
    for arg in a.driver_arg:
        if not re.fullmatch(r"--[A-Za-z0-9_-]+=.*", arg, flags=re.S):
            p.error("--driver-arg requires one --key=value argument")
        key, value = arg.split("=", 1)
        key = key.replace("_", "-")
        if key in ("--output-dir", "--out-dir"):
            p.error("output directory is managed by this launcher")
        if key == "--strategy":
            p.error("use --strategies instead of overriding the managed --strategy argument")
        overrides[key] = value
    input_identity = None
    if "--input" in overrides:
        input_path = Path(overrides["--input"])
        input_path = input_path.resolve() if input_path.is_absolute() else (ROOT / input_path).resolve()
        if not input_path.is_file():
            p.error("input file does not exist: " + str(input_path))
        overrides["--input"] = str(input_path)
        input_identity = {"path_on_launcher_host": str(input_path), "bytes": input_path.stat().st_size,
                          "sha256": sha(input_path), "other_node_hashes": "must be checked during staging"}
    preflight = None
    runtime_env = {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1")}
    if a.mode == "local-smoke":
        runtime_env["CUDA_VISIBLE_DEVICES"] = a.physical_gpu
        if not a.prepare_only:
            probe = subprocess.run(["nvidia-smi", "-i", a.physical_gpu,
                                    "--query-gpu=index,uuid,name,memory.free", "--format=csv,noheader,nounits"],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            preflight = {"checked_at_utc": now(), "query": probe.stdout.strip(), "continuous_monitoring": False}
            free_mib = float(probe.stdout.strip().split(",")[-1])
            minimum_free_mib = max(a.min_free_mib, 512 * max(ranks) + 512)
            if free_mib < minimum_free_mib:
                p.error(f"one-time memory preflight: {free_mib} MiB free, below context/array budget {minimum_free_mib} MiB")
    else:
        visibility = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(str(i) for i in range(a.gpus_per_node)))
        tokens = [x.strip() for x in visibility.split(",")]
        normalized = [str(int(x)) if x.isdecimal() else x for x in tokens]
        if any(not x for x in normalized) or len(set(normalized)) != len(normalized):
            p.error("CUDA_VISIBLE_DEVICES must contain distinct nonempty GPU identifiers")
        if len(normalized) < a.gpus_per_node:
            p.error("CUDA_VISIBLE_DEVICES exposes fewer GPUs than --gpus-per-node requests")
        runtime_env["CUDA_VISIBLE_DEVICES"] = ",".join(tokens)
    run_dir = a.run_dir.resolve()
    if run_dir.exists():
        p.error("refusing to overwrite or duplicate existing run: " + str(run_dir))
    steps = []
    for nranks in ranks:
        for mode, strategy in [(m, s) for m in modes for s in strategies]:
            node_count = nranks // a.gpus_per_node if a.mode == "cluster" else 1
            name = (f"nodes{node_count}_gpn{a.gpus_per_node}_ranks{nranks}_{mode}" if a.mode == "cluster"
                    else f"ranks{nranks}_{mode}") + "_" + strategy
            command = [launcher, "-np" if openmpi else "-n", str(nranks)]
            if a.mode == "cluster":
                command += ["--hostfile" if openmpi else "-f", str(a.hostfile.resolve())]
                if openmpi:
                    command += ["--map-by", f"ppr:{a.gpus_per_node}:node", "--bind-to", "none"]
                else:
                    command += ["-ppn", str(a.gpus_per_node)]
            elif openmpi:
                command += ["--oversubscribe", "--bind-to", "none"]
            if openmpi:
                for key in ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS"):
                    if key in env or key in runtime_env:
                        command += ["-x", key]
            command += a.mpi_arg + [str(binary)]
            parameters = {"--nside": "16", "--synthetic": "1", "--day-count": "1", "--segs": "2",
                          "--time-count": "8", "--tlen-limit": "4", "--segments-limit": "2",
                          "--serial-check": "1", "--dump-validation": "1", "--recon-mode": mode,
                          "--strategy": strategy}
            if a.mode == "local-smoke":
                parameters["--allow-shared-gpu"] = "1"
            if "--input" in overrides and "--synthetic" not in overrides:
                parameters["--synthetic"] = "0"
            parameters.update(overrides)
            command += [key + "=" + value for key, value in parameters.items()]
            command += ["--output-dir=" + str(run_dir / name / "output")]
            steps.append({"name": name, "node_count": node_count, "gpus_per_node": a.gpus_per_node,
                          "ranks": nranks, "recon_mode": mode, "strategy": strategy, "command": command})
    run_dir.mkdir(parents=True)
    config = {"schema": "cluster_mpi_queue_v2", "state": "planned" if a.prepare_only else "submitted",
              "prepare_only": a.prepare_only, "run_dir": str(run_dir), "mode": a.mode,
              "gpus_per_node": a.gpus_per_node,
              "strategies": strategies,
              "planned_matrix": [{"node_count": n // a.gpus_per_node, "gpus_per_node": a.gpus_per_node, "ranks": n}
                                 for n in ranks] if a.mode == "cluster" else None,
              "created_at_utc": now(), "timeout_s": a.timeout_s, "environment": runtime_env,
              "binary": str(binary), "binary_sha256": sha(binary) if binary.is_file() else None,
              "mpi_flavor": "openmpi" if openmpi else "mpich", "mpi_version": version,
              "input_identity": input_identity,
              "hostfile": {"path": str(a.hostfile.resolve()), "sha256": sha(a.hostfile)} if a.hostfile else None,
              "mpi_transport_environment": {key: env[key] for key in (
                  "MPIR_CVAR_CH4_NETMOD", "FI_PROVIDER", "UCX_TLS", "UCX_NET_DEVICES",
                  "OMPI_MCA_btl_tcp_if_include", "OMPI_MCA_oob_tcp_if_include", "MPIEXEC_PORT_RANGE", "HYDRA_IFACE") if key in env},
              "one_time_memory_preflight": preflight, "steps": steps,
              "cross_node_results_claimed": False, "continuous_resource_monitoring": False,
              "scope": ("Planned commands only; no MPI, SMI, worker or GPU execution." if a.prepare_only else
                        "Local smoke uses one physical GPU; cluster hostname count requires provider-instance verification. No performance/scaling claim is inferred.")}
    config_path = run_dir / "launch_manifest.json"
    write_json(config_path, config)
    if a.prepare_only:
        write_json(run_dir / "status.json", {"state": "planned", "prepare_only": True,
                                            "gpu_execution": False, "continuous_resource_monitoring": False})
        print(json.dumps({"state": "planned", "run_dir": str(run_dir), "manifest": str(config_path), "steps": len(steps)}, indent=2))
        return 0
    (run_dir / ".queue.lock").mkdir()
    write_json(run_dir / "status.json", {"state": "submitted", "continuous_resource_monitoring": False})
    if a.foreground:
        return worker(config_path)
    with (run_dir / "background.log").open("a") as log:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)],
                                   cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    ack = {"submitted_at_utc": now(), "pid": process.pid, "run_dir": str(run_dir),
           "status_file": str(run_dir / "status.json"), "continuous_resource_monitoring": False}
    write_json(run_dir / "launch_ack.json", ack)
    print(json.dumps(ack, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
