#!/usr/bin/env python3
"""Build MPI/CUDA without assuming Open MPI, system MPI, or administrator access."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    env = os.environ.copy()
    dep = ROOT / "deps/mpich"
    if (dep / "bin").is_dir():
        env["PATH"] = str(dep / "bin") + os.pathsep + env.get("PATH", "")
        env["LD_LIBRARY_PATH"] = str(dep / "lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


def capture(argv, env):
    p = subprocess.run(argv, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "build/cluster_sts_mpi")
    p.add_argument("--archs", default=os.environ.get("CUDA_ARCHS", "86;89"))
    p.add_argument("--nvcc", default=os.environ.get("NVCC"))
    p.add_argument("--mpicxx", default=os.environ.get("MPICXX"))
    p.add_argument("--no-fast-math", action="store_true")
    p.add_argument("--skip-reference", action="store_true", help="Skip compiling the unchanged production reference")
    a = p.parse_args()
    env = environment()
    nvcc = a.nvcc or shutil.which("nvcc", path=env["PATH"])
    if not nvcc and Path("/usr/local/cuda-12.4/bin/nvcc").exists():
        nvcc = "/usr/local/cuda-12.4/bin/nvcc"
    mpicxx = a.mpicxx or shutil.which("mpicxx", path=env["PATH"]) or shutil.which("mpic++", path=env["PATH"])
    if not nvcc or not mpicxx:
        p.error("nvcc and mpicxx are required; set NVCC/MPICXX or see README for an isolated user MPI dependency")
    architectures = [x for x in re.split(r"[;, ]+", a.archs) if x]
    if not architectures or any(not re.fullmatch(r"[0-9]{2,3}", x) for x in architectures):
        p.error("--archs must contain architecture numbers, e.g. '86;89'")
    rc1, compile_flags, _ = capture([mpicxx, "--showme:compile"], env)
    rc2, link_flags, _ = capture([mpicxx, "--showme:link"], env)
    if rc1 == 0 and rc2 == 0:
        mpi_flags = shlex.split(compile_flags) + shlex.split(link_flags)
        wrapper_style = "openmpi_showme"
    else:
        rc, shown, err = capture([mpicxx, "-show"], env)
        if rc:
            raise RuntimeError("Cannot query MPI wrapper flags: " + err)
        words = shlex.split(shown)
        if len(words) < 2:
            raise RuntimeError("Unexpected MPI wrapper -show output: " + shown)
        mpi_flags = words[1:]
        wrapper_style = "mpich_show"
    # nvcc accepts include/library flags; host driver flags need explicit forwarding.
    forwarded = []
    linker_tokens = []
    for flag in mpi_flags:
        if flag.startswith(("-I", "-L", "-l", "-D", "-U")) or flag.endswith((".a", ".so")):
            forwarded.append(flag)
        elif flag.startswith("-Wl,"):
            linker_tokens.extend(flag[4:].split(","))
        elif flag in ("-pthread", "-fopenmp"):
            forwarded.append("-Xcompiler=" + flag)
        else:
            forwarded.append(flag)
    # Do not encode an old package root as the sole MPI search location.
    i = 0
    while i < len(linker_tokens):
        flag = linker_tokens[i]
        if flag in ("-rpath", "--rpath") and i + 1 < len(linker_tokens):
            value = linker_tokens[i + 1]
            if not value.startswith(str(ROOT)):
                forwarded += ["-Xlinker=" + flag, "-Xlinker=" + value]
            i += 2
            continue
        if flag.startswith(("-rpath=", "--rpath=")) and flag.split("=", 1)[1].startswith(str(ROOT)):
            i += 1
            continue
        forwarded.append("-Xlinker=" + flag)
        i += 1
    forwarded += ["-Xlinker=-rpath", "-Xlinker=$ORIGIN/../deps/mpich/lib"]
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [nvcc, "-O3", "-lineinfo", "-std=c++17", "-Xcompiler=-fopenmp"]
    if not a.no_fast_math:
        command.append("--use_fast_math")
    if os.environ.get("CXX"):
        command += ["-ccbin", os.environ["CXX"]]
    for arch in architectures:
        command.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
    # Retain a PTX fallback; the exact SASS targets above remain preferred.
    command.append(f"-gencode=arch=compute_{architectures[0]},code=compute_{architectures[0]}")
    command += ["-I" + str(ROOT / "vendor/final_sts"), str(ROOT / "src/main_cluster_mpi.cu")]
    command += forwarded + ["-o", str(output)]
    log = output.parent / "build.log"
    print(shlex.join(command), flush=True)
    with log.open("w") as f:
        f.write(shlex.join(command) + "\n")
        f.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    reference = output.parent / "production_reference"
    reference_command = None
    reference_rc = None
    reference_source = ROOT / "vendor/final_sts/main_recon_online_mirror_tile_vissfast.cu"
    if result.returncode == 0 and not a.skip_reference and reference_source.is_file():
        reference_command = [str(reference_source) if x == str(ROOT / "src/main_cluster_mpi.cu") else x for x in command]
        reference_command[-1] = str(reference)
        with (output.parent / "reference_build.log").open("w") as f:
            f.write(shlex.join(reference_command) + "\n")
            f.flush()
            reference_rc = subprocess.run(reference_command, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    source_files = []
    for directory in ("src", "vendor", "scripts"):
        for f in sorted((ROOT / directory).rglob("*")):
            if f.is_file() and "__pycache__" not in f.parts:
                source_files.append({"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": sha(f)})
    manifest = {
        "schema": "cluster_mpi_build_v1", "built_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command, "cwd": str(ROOT), "exit_code": result.returncode,
        "cuda_architectures": architectures, "fast_math": not a.no_fast_math,
        "mpi_wrapper": mpicxx, "mpi_wrapper_style": wrapper_style, "mpi_flags": mpi_flags,
        "nvcc_version": capture([nvcc, "--version"], env)[1],
        "sources": source_files, "binary": str(output),
        "binary_sha256": sha(output) if result.returncode == 0 else None,
        "reference_command": reference_command, "reference_exit_code": reference_rc,
        "reference_binary": str(reference) if reference_rc == 0 else None,
        "reference_binary_sha256": sha(reference) if reference_rc == 0 else None,
        "build_log": str(log), "scope": "Build only; no GPU or cross-node execution is implied."
    }
    (output.parent / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if result.returncode:
        print(log.read_text()[-18000:], file=sys.stderr)
    return result.returncode or (reference_rc or 0)


if __name__ == "__main__":
    sys.exit(main())
