#!/usr/bin/env python3
"""Unpack a verified MPICH wheel into a relocatable prefix (including .data/data/lib)."""
import argparse
import base64
import csv
import datetime as dt
from email import message_from_string
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relative_target(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("unsafe wheel member: " + name)
    parts = path.parts
    if parts and parts[0].endswith(".data"):
        if len(parts) < 3:
            raise ValueError("invalid wheel .data entry: " + name)
        scheme, tail = parts[1], parts[2:]
        if scheme in ("data", "purelib", "platlib"):
            return Path(*tail)
        if scheme == "scripts":
            return Path("bin", *tail)
        if scheme == "headers":
            return Path("include", *tail)
        raise ValueError("unsupported wheel install scheme: " + scheme)
    return Path(*parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--prefix", type=Path, default=ROOT / "deps/mpich")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    prefix = args.prefix.resolve()
    if prefix.exists():
        parser.error("refusing to overwrite existing prefix: " + str(prefix))
    wheel_hash = digest(wheel)
    if args.expected_sha256 and wheel_hash.lower() != args.expected_sha256.lower():
        parser.error("wheel SHA256 mismatch")
    staging = prefix.with_name(prefix.name + ".installing." + str(os.getpid()))
    if staging.exists():
        parser.error("staging directory already exists")
    records = []
    try:
        staging.mkdir(parents=True)
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            record_files = [x for x in names if x.endswith(".dist-info/RECORD")]
            metadata_files = [x for x in names if x.endswith(".dist-info/METADATA")]
            if len(record_files) != 1 or len(metadata_files) != 1:
                raise ValueError("wheel metadata is ambiguous")
            metadata = message_from_string(archive.read(metadata_files[0]).decode())
            if metadata.get("Name", "").lower() != "mpich" or metadata.get("Version") != "5.0.2":
                raise ValueError("this pinned bootstrap expects the mpich==5.0.2 wheel")
            wheel_records = {row[0]: row[1:] for row in csv.reader(io.StringIO(archive.read(record_files[0]).decode()))}
            targets = set()
            symlinks = []
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if info.filename not in wheel_records:
                    raise ValueError("member is missing from RECORD: " + info.filename)
                data = archive.read(info)
                record_hash, record_size = wheel_records[info.filename]
                if record_size and len(data) != int(record_size):
                    raise ValueError("RECORD size mismatch: " + info.filename)
                if record_hash:
                    algorithm, encoded = record_hash.split("=", 1)
                    expected = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
                    if hashlib.new(algorithm, data).digest() != expected:
                        raise ValueError("RECORD digest mismatch: " + info.filename)
                target = relative_target(info.filename)
                if str(target) in targets:
                    raise ValueError("wheel entries collide at prefix path: " + str(target))
                targets.add(str(target))
                installed = staging / target
                installed.parent.mkdir(parents=True, exist_ok=True)
                mode = (info.external_attr >> 16) & 0o777
                is_link = stat.S_ISLNK(info.external_attr >> 16)
                if is_link:
                    link = data.decode()
                    if Path(link).is_absolute() or not (installed.parent / link).resolve().is_relative_to(staging):
                        raise ValueError("unsafe wheel symlink: " + info.filename)
                    symlinks.append((installed, link))
                else:
                    installed.write_bytes(data)
                    installed.chmod(mode or (0o755 if target.parts[0] == "bin" else 0o644))
                records.append({"wheel_member": info.filename, "prefix_path": str(target), "bytes": len(data),
                                "sha256": hashlib.sha256(data).hexdigest(), "record_hash_verified": bool(record_hash),
                                "mode": oct(mode), "kind": "symlink" if is_link else "file"})
            for installed, link in symlinks:
                installed.symlink_to(link)
            for needed in ("bin/mpicxx", "bin/mpiexec", "include/mpi.h"):
                if not (staging / needed).is_file():
                    raise ValueError("required prefix artifact missing: " + needed)
            if not list((staging / "lib").glob("libmpi.so*")):
                raise ValueError("libmpi.so is missing; do not use a partial pip --target layout")
        manifest = {"schema": "mpich_wheel_prefix_layout_v1", "version": "5.0.2",
                    "installed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "wheel": str(wheel), "wheel_sha256": wheel_hash,
                    "external_expected_sha256_matched": bool(args.expected_sha256),
                    "prefix": str(prefix), "record_verification": "all hashed members verified",
                    "note": ".data/data members were mapped into prefix; wheel license metadata was retained.",
                    "files": records}
        (staging / "PREFIX_LAYOUT.json").write_text(json.dumps(manifest, indent=2) + "\n")
        staging.replace(prefix)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps({"prefix": str(prefix), "wheel_sha256": wheel_hash, "files": len(records)}, indent=2))


if __name__ == "__main__":
    main()
