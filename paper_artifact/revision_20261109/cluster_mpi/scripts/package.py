#!/usr/bin/env python3
"""Create a portable source archive, source manifest, and archive SHA256."""
import argparse
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--version", default=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    p.add_argument("--output-dir", type=Path, default=ROOT / "releases")
    p.add_argument("--include-binary", action="store_true", help="Optional same-stack convenience binary; recompile on rented nodes")
    p.add_argument("--with-runtime", action="store_true", help="Bundle build outputs and isolated deps/mpich, including its license metadata")
    a = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.version):
        p.error("version must be a filename-safe token")
    files = []
    for directory in ("src", "vendor", "scripts", "config", "tests", "docs"):
        for f in sorted((ROOT / directory).rglob("*")):
            if f.is_file() and not f.is_symlink() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo"):
                files.append(f)
    for name in ("README.md", "VALIDATION.md", "CHANGELOG.md", "ORIGIN.json"):
        if (ROOT / name).is_file():
            files.append(ROOT / name)
    include_binary = a.include_binary or a.with_runtime
    if include_binary:
        for name in ("cluster_sts_mpi", "build_manifest.json", "build.log"):
            file = ROOT / "build" / name
            if not file.is_file():
                p.error("--include-binary requires successful build and manifest: " + str(file))
            files.append(file)
        build_manifest = json.loads((ROOT / "build/build_manifest.json").read_text())
        if build_manifest.get("exit_code") != 0 or build_manifest.get("binary_sha256") != sha(ROOT / "build/cluster_sts_mpi"):
            p.error("build identity is not valid")
        for entry in build_manifest.get("sources", []):
            if entry["path"].startswith(("src/", "vendor/")):
                actual = ROOT / entry["path"]
                if not actual.is_file() or sha(actual) != entry["sha256"]:
                    p.error("binary source changed since build; rebuild before packaging: " + entry["path"])
        reference = ROOT / "build/production_reference"
        if (build_manifest.get("reference_exit_code") == 0 and reference.is_file()
                and build_manifest.get("reference_binary_sha256") == sha(reference)):
            files.append(reference)
            if (ROOT / "build/reference_build.log").is_file():
                files.append(ROOT / "build/reference_build.log")
    if a.with_runtime:
        dependency = ROOT / "deps/mpich"
        if not dependency.is_dir():
            p.error("--with-runtime requires isolated deps/mpich (see README)")
        for file in sorted(dependency.rglob("*")):
            if file.is_file() and "__pycache__" not in file.parts and file.suffix not in (".pyc", ".pyo"):
                if file.is_symlink() and not file.resolve().is_relative_to(ROOT):
                    p.error("external dependency symlink is not portable: " + str(file))
                files.append(file)
    files = sorted(set(files))
    if not (ROOT / "src/main_cluster_mpi.cu").is_file():
        p.error("cluster source is missing")
    manifest = {
        "schema": "cluster_mpi_source_package_v1", "version": a.version,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "includes_binary": include_binary, "includes_mpich_runtime": a.with_runtime,
        "runtime_scope": "Optional runtime is Linux x86_64; rented OS, CUDA driver and network still need validation. Source rebuild is authoritative.",
        "excludes": ["input sky/maps", "run outputs", "unrelated dependencies", "SSH configuration/keys"],
        "files": [{"path": str(f.relative_to(ROOT)), "bytes": f.stat().st_size, "sha256": sha(f),
                   "kind": "symlink" if f.is_symlink() else "file",
                   "link_target": os.readlink(f) if f.is_symlink() else None} for f in files]
    }
    a.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_linux-x86_64-runtime" if a.with_runtime else ("_binary" if a.include_binary else "_source")
    stem = "cluster_mpi_" + a.version + suffix
    archive = a.output_dir / (stem + ".tar.gz")
    if archive.exists():
        p.error("refusing to overwrite existing release: " + str(archive))
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    with tarfile.open(archive, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=stem + "/" + str(f.relative_to(ROOT)), recursive=False)
        info = tarfile.TarInfo(stem + "/PACKAGE_MANIFEST.json")
        info.size = len(data)
        info.mode = 0o644
        tar.addfile(info, io.BytesIO(data))
    digest = sha(archive)
    archive.with_suffix(archive.suffix + ".sha256").write_text(digest + "  " + archive.name + "\n")
    archive.with_suffix(archive.suffix + ".manifest.json").write_bytes(data)
    print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest, "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
