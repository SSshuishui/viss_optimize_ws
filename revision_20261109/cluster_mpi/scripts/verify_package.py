#!/usr/bin/env python3
"""Verify an extracted source/runtime archive against its embedded file manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text())
    failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.resolve().is_relative_to(root):
            failures.append({"path": entry["path"], "reason": "path escapes package"})
        elif not path.is_file():
            failures.append({"path": entry["path"], "reason": "missing"})
        elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            failures.append({"path": entry["path"], "reason": "size or SHA256 mismatch"})
        elif entry.get("kind") == "symlink" and (not path.is_symlink() or os.readlink(path) != entry["link_target"]):
            failures.append({"path": entry["path"], "reason": "symlink target mismatch"})
    result = {"state": "passed" if not failures else "failed", "files": len(manifest["files"]), "failures": failures,
              "scope": "File identity only; does not prove CUDA/MPI runtime or cross-node correctness."}
    print(json.dumps(result, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
