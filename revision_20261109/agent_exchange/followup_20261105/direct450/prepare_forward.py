#!/usr/bin/env python3
"""Derive the frozen cache exporter with UUID-only device selection."""

import hashlib
import sys
from pathlib import Path

src, out = (Path(arg) for arg in sys.argv[1:])
data = src.read_bytes()
expected = "e382364cc0d34302b6d9537e28056ffa65b3d0ecfb8d02e51b785c63c2461e50"
if hashlib.sha256(data).hexdigest() != expected:
    raise SystemExit("Unexpected frozen forward source")
old = b"  auto visible_gpus = parse_gpus(gpus_s);\n  int requested_G = (int)visible_gpus.size();"
new = (
    b'  if(gpus_s.rfind("GPU-",0)!=0 || gpus_s.find(\',\')!=std::string::npos)\n'
    b'    throw std::runtime_error("pass exactly one GPU UUID");\n'
    b"  int requested_G = 1;"
)
if data.count(old) != 1:
    raise SystemExit("GPU-selector patch anchor is not unique")
old_count = b"  if(devCount <= 0){"
if data.count(old_count) != 1:
    raise SystemExit("GPU-count patch anchor is not unique")
with out.open("xb") as handle:
    handle.write(data.replace(old, new).replace(old_count, b"  if(devCount != 1){"))
