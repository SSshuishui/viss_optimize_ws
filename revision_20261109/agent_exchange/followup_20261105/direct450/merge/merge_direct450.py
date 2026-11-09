#!/usr/bin/env python3
"""Validate, pool, and compare committed lanes of the 10 MHz direct 450-day run."""

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

from pair_stats import REGIONS, Moments, finalize, native_latitude_tags


NSIDE = 4096
DAY_IDS = tuple(range(1, 451))
SEGMENTS_PER_DAY = 10
RECORDS_PER_SEGMENT = 195_468
RECORDS_PER_DAY = SEGMENTS_PER_DAY * RECORDS_PER_SEGMENT
CACHE_BYTES = 36 + RECORDS_PER_SEGMENT * 15 * 4
SKY_SHA256 = "277b6cb7299f2e8e54cc1758a045e7589686e28dc79a1df2f56aba5fc2d2d5ac"
PROTOCOL_SHA256 = "641049fa7a45dcab3b38135c5c3b7ccc2ddce6652b511966c2dc234c49272ded"
CURRENT_SCHEMA = "direct450_lane_current_v1"
GENERATION_SCHEMA = "direct450_lane_generation_v1"


def sha_text(value, context):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Invalid SHA256 for {context}")
    return value


def canonical_sha(obj):
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def local_child(root, value):
    name = Path(value)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Lane path must be relative and self-contained: {value}")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Lane path escapes its archive: {value}")
    return target


def day_list(value, context):
    if (not isinstance(value, list) or not value or
            any(type(day) is not int for day in value) or value != sorted(set(value))):
        raise ValueError(f"Expected sorted distinct integer day IDs in {context}")
    return tuple(value)


@dataclass(frozen=True)
class Array:
    path: Path
    dtype: str
    pixels: int
    sha256: str

    @property
    def bytes(self):
        return self.pixels * np.dtype(self.dtype).itemsize

    def validate_size(self):
        if not self.path.is_file() or self.path.stat().st_size != self.bytes:
            raise ValueError(f"Unexpected raw array byte count: {self.path} (expected {self.bytes})")

    def record(self):
        return {"path": str(self.path), "dtype": self.dtype, "count": self.pixels,
                "bytes": self.bytes, "sha256": self.sha256}


def array_from_record(path, record, dtype, pixels):
    result = Array(path.resolve(), dtype, pixels, sha_text(record.get("sha256"), path))
    if (record.get("dtype") != dtype or record.get("count", pixels) != pixels or
            record.get("bytes") != result.bytes):
        raise ValueError(f"Wrong array dtype, count or byte count in manifest for {path}")
    result.validate_size()
    return result


class ArrayReader:
    def __init__(self, array, stack):
        self.array = array
        self.stream = stack.enter_context(array.path.open("rb"))
        self.itemsize = np.dtype(array.dtype).itemsize
        self.sha = hashlib.sha256()

    def read(self, count):
        raw = self.stream.read(count * self.itemsize)
        if len(raw) != count * self.itemsize:
            raise ValueError(f"Short read: {self.array.path}")
        self.sha.update(raw)
        return np.frombuffer(raw, dtype=self.array.dtype)

    def finish(self):
        if self.stream.read(1) or self.sha.hexdigest() != self.array.sha256:
            raise ValueError(f"Trailing bytes or SHA256 mismatch: {self.array.path}")


class ArrayWriter:
    def __init__(self, path, dtype, stack):
        self.path = path
        self.dtype = dtype
        self.stream = stack.enter_context(path.open("xb"))
        self.sha = hashlib.sha256()
        self.pixels = 0

    def write(self, values):
        raw = values.astype(self.dtype, copy=False).tobytes()
        self.stream.write(raw)
        self.sha.update(raw)
        self.pixels += values.size

    def finish(self, pixels):
        if self.pixels != pixels:
            raise ValueError(f"Output count mismatch: {self.path}")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        return {"path": self.path.name, "dtype": self.dtype, "count": pixels,
                "bytes": self.path.stat().st_size, "sha256": self.sha.hexdigest()}


def verify_day_receipt(record, pixels, path):
    day = record["day"]
    context = f"{path} day {day}"
    if (record.get("status") != "accepted" or record.get("records") != RECORDS_PER_DAY or
            record.get("segments") != SEGMENTS_PER_DAY):
        raise ValueError(f"Invalid day acceptance receipt: {context}")
    for key in ("visibility_sha256", "geometry_sha256"):
        sha_text(record.get(key), f"{context} {key}")
    segments = record["segment_receipts"]
    if tuple(entry.get("segment") for entry in segments) != tuple(range(1, SEGMENTS_PER_DAY + 1)):
        raise ValueError(f"Missing or reordered segment acceptance receipts: {context}")
    for entry in segments:
        segment = entry["segment"]
        if (entry.get("records") != RECORDS_PER_SEGMENT or entry.get("bytes") != CACHE_BYTES or
                entry.get("name") != f"day{day}_seg{segment}.bin"):
            raise ValueError(f"Invalid segment receipt: {context} segment {segment}")
        for key in ("sha256", "visibility_sha256", "geometry_sha256"):
            sha_text(entry.get(key), f"{context} segment {segment} {key}")
    for key, name, dtype in (("rawN", "raw_N.f32", "<f4"), ("W", "W.f64", "<f8")):
        entry = record["direct_arrays"][key]
        if (entry.get("file") != name or entry.get("format") != dtype or
                entry.get("count") != pixels or entry.get("finite_count") != pixels or
                entry.get("nonfinite") != 0 or entry.get("bytes") != pixels * np.dtype(dtype).itemsize):
            raise ValueError(f"Invalid daily direct output receipt: {context} {key}")
        sha_text(entry.get("sha256"), f"{context} {key}")


def read_lane(path, pixels):
    path = path.resolve()
    if path.is_dir():
        path /= "CURRENT.json"
    manifest, manifest_sha = load_json(path)
    current = None
    if manifest.get("schema") == CURRENT_SCHEMA:
        current = manifest
        path = local_child(path.parent, current["manifest_path"])
        manifest, manifest_sha = load_json(path)
        if manifest_sha != sha_text(current.get("manifest_sha256"), path):
            raise ValueError(f"CURRENT manifest SHA mismatch: {path}")
    if manifest.get("schema") != GENERATION_SCHEMA or manifest.get("status") != "committed":
        raise ValueError(f"Expected a committed lane generation: {path}")
    assigned = day_list(manifest.get("assigned_days"), path)
    completed = day_list(manifest.get("completed_days"), path)
    if assigned != completed:
        raise ValueError(f"Partial lane cannot enter final merge: {path}")
    if current is not None and day_list(current.get("completed_days"), path) != completed:
        raise ValueError(f"CURRENT and committed generation disagree: {path}")

    receipts = manifest.get("days")
    if not isinstance(receipts, list) or tuple(record.get("day") for record in receipts) != completed:
        raise ValueError(f"Day acceptance receipts do not match committed days: {path}")
    for record in receipts:
        verify_day_receipt(record, pixels, path)

    protocol = manifest.get("protocol")
    protocol_sha = sha_text(manifest.get("protocol_sha256"), f"{path} protocol")
    if not isinstance(protocol, dict) or canonical_sha(protocol) != protocol_sha:
        raise ValueError(f"Lane science protocol fingerprint mismatch: {path}")
    identity = manifest["lane"]
    if (not identity.get("lane_id") or not identity.get("host") or not identity.get("gpu") or
            day_list(identity.get("assigned_days"), path) != assigned):
        raise ValueError(f"Lane hardware and build identity missing: {path}")
    sha_text(identity.get("runner_sha256"), f"{path} runner")
    for key in ("sky", "dcf", "forward_bin", "direct_bin"):
        receipt = identity["inputs"][key]
        sha_text(receipt.get("sha256"), f"{path} {key}")
        if type(receipt.get("bytes")) is not int or receipt["bytes"] <= 0:
            raise ValueError(f"Invalid lane input size: {path} {key}")
        if key in ("sky", "dcf") and any(receipt.get(field) != value for field, value in protocol[key].items()):
            raise ValueError(f"Lane input differs from shared science protocol: {path} {key}")
    arrays = {}
    for key in ("N", "W"):
        entry = manifest["arrays"][key]
        array_path = local_child(path.parent, entry["path"])
        arrays[key] = array_from_record(array_path, entry, "<f8", pixels)
    if arrays["N"].path == arrays["W"].path:
        raise ValueError(f"Lane numerator and denominator alias: {path}")
    return {"path": str(path), "manifest_sha256": manifest_sha, "days": completed,
            "protocol": protocol, "protocol_sha256": protocol_sha, "lane": manifest["lane"], "arrays": arrays}


def validate_lanes(lanes, expected_days):
    coverage = set()
    array_paths = set()
    for lane in lanes:
        overlap = coverage.intersection(lane["days"])
        if overlap:
            raise ValueError(f"Duplicate lane days: {sorted(overlap)}")
        coverage.update(lane["days"])
        for array in lane["arrays"].values():
            if array.path in array_paths:
                raise ValueError(f"Two lanes reuse one accumulator: {array.path}")
            array_paths.add(array.path)
        if lane["protocol_sha256"] != lanes[0]["protocol_sha256"]:
            raise ValueError(f"Lane science protocols differ: {lane['path']}")
    missing = set(expected_days) - coverage
    unexpected = coverage - set(expected_days)
    if missing or unexpected:
        raise ValueError(f"Invalid day coverage. Missing={sorted(missing)} unexpected={sorted(unexpected)}")


def historical_arrays(args, nside, pixels, expected_days):
    archive, archive_sha = load_json(args.sts_manifest)
    if (archive.get("status") != "verified" or archive.get("nside") != nside or
            archive.get("observed_days") != [expected_days[0], expected_days[-1]] or
            archive.get("segments_per_day") != SEGMENTS_PER_DAY or archive.get("log_exit") != 0):
        raise ValueError("STS archive is not a verified pooled C/W result over the target day span")
    sha_text(archive.get("log_sha256"), "STS archive log")
    arrays = {"input": Array(args.input.resolve(), "<f4", pixels, args.input_sha)}
    for name, path, dtype in (("C", args.sts, "<f4"), ("W", args.sts_w, "<u4")):
        record = archive["files"][name]
        if record.get("finite_pixels") != pixels or record.get("zero_pixels") != 0:
            raise ValueError(f"Expected every historical STS {name} pixel valid and nonzero")
        arrays["sts_" + name] = array_from_record(path, record, dtype, pixels)
    if args.regions is not None:
        arrays["regions"] = Array(args.regions.resolve(), "u1", pixels, args.regions_sha)
    for array in arrays.values():
        array.validate_size()
    return arrays, archive_sha


def add_comparison(moments, reference, candidate, masks):
    moments[0].add(reference, candidate)
    for mask, moment in zip(masks, moments[1:]):
        moment.add(reference[mask], candidate[mask])


COMPARISONS = {
    "sts_vs_dcf_pooled": ("corrected-DCF direct pooled N/W", "archived original STS pooled C"),
    "sts_vs_dcf_raw_n": ("corrected-DCF direct raw N summed over all days", "archived original STS pooled C"),
    "direct_vs_input": ("frozen input sky B", "corrected-DCF direct pooled N/W"),
    "sts_vs_input": ("frozen input sky B", "archived original STS pooled C"),
}


def merge(args, nside=NSIDE, expected_days=DAY_IDS, expected_protocol_sha=PROTOCOL_SHA256):
    pixels = 12 * nside**2
    lanes = sorted((read_lane(path, pixels) for path in args.lanes), key=lambda lane: lane["days"][0])
    validate_lanes(lanes, expected_days)
    protocol_sha = lanes[0]["protocol_sha256"]
    if protocol_sha != expected_protocol_sha or (args.protocol_sha and args.protocol_sha != protocol_sha):
        raise ValueError("Lane protocol differs from the requested run fingerprint")
    if lanes[0]["protocol"]["sky"] != {"bytes": 4 * pixels, "sha256": args.input_sha}:
        raise ValueError("Input sky identity differs from the lane protocol")
    input_arrays, archive_sha = historical_arrays(args, nside, pixels, expected_days)
    target = args.output.resolve()
    if target.exists() or not target.parent.is_dir():
        raise ValueError("Output directory must be new and its parent directory must exist")
    required = pixels * (4 if args.pooled_only else 20)
    if shutil.disk_usage(target.parent).free < required + min(1 << 30, required):
        raise ValueError(f"Insufficient merge output disk space for {required} array bytes plus margin")

    moments = {name: [Moments() for _ in REGIONS] for name in COMPARISONS}
    extrema = {key: [float("inf"), float("-inf")] for key in ("N", "W", "C")}
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as directory:
        staging = Path(directory)
        with ExitStack() as stack:
            lane_readers = [{key: ArrayReader(array, stack) for key, array in lane["arrays"].items()} for lane in lanes]
            reads = {key: ArrayReader(array, stack) for key, array in input_arrays.items()}
            writes = {"C": ArrayWriter(staging / "direct_C.f32", "<f4", stack)}
            if not args.pooled_only:
                writes.update({key: ArrayWriter(staging / f"direct_{key}.f64", "<f8", stack) for key in ("N", "W")})

            for start in range(0, pixels, args.chunk):
                count = min(args.chunk, pixels - start)
                n = np.zeros(count, dtype=np.float64)
                w = np.zeros(count, dtype=np.float64)
                for lane, readers in zip(lanes, lane_readers):
                    numerator = readers["N"].read(count)
                    weight = readers["W"].read(count)
                    if (not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(weight)) or
                            np.any(weight < 0) or np.any((weight == 0) & (numerator != 0))):
                        raise ValueError(f"Invalid lane N/W in {lane['path']} at pixel {start}")
                    n += numerator
                    w += weight
                if not np.all(np.isfinite(n)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
                    raise ValueError(f"Invalid pooled direct response at pixel {start}")
                c = (n / w).astype("<f4")
                sky = reads["input"].read(count).astype(np.float64)
                sts = reads["sts_C"].read(count).astype(np.float64)
                sts_w = reads["sts_W"].read(count)
                if (not np.all(np.isfinite(c)) or not np.all(np.isfinite(sky)) or
                        not np.all(np.isfinite(sts)) or np.any(sts_w == 0) or np.any(sts_w > len(expected_days) * RECORDS_PER_DAY)):
                    raise ValueError(f"Invalid sky or STS/direct output at pixel {start}")
                tags = (reads["regions"].read(count) if "regions" in reads else
                        native_latitude_tags(nside, start, start + count))
                if np.any(tags > 2):
                    raise ValueError(f"Unexpected native latitude tag at pixel {start}")
                masks = [tags == i for i in range(3)]
                direct = c.astype(np.float64)
                add_comparison(moments["sts_vs_dcf_pooled"], direct, sts, masks)
                add_comparison(moments["sts_vs_dcf_raw_n"], n, sts, masks)
                add_comparison(moments["direct_vs_input"], sky, direct, masks)
                add_comparison(moments["sts_vs_input"], sky, sts, masks)
                for key, values in (("N", n), ("W", w), ("C", c)):
                    extrema[key][0] = min(extrema[key][0], float(values.min()))
                    extrema[key][1] = max(extrema[key][1], float(values.max()))
                    if key in writes:
                        writes[key].write(values)
                if start == 0 or (start // args.chunk + 1) % 16 == 0 or start + count == pixels:
                    print(f"Verified and pooled {start + count:,}/{pixels:,} pixels", flush=True)

            for readers in [*lane_readers, reads]:
                for reader in readers.values():
                    reader.finish()
            outputs = {key: writer.finish(pixels) for key, writer in writes.items()}

        metrics = {name: {"reference": COMPARISONS[name][0], "candidate": COMPARISONS[name][1],
                          **finalize(values, pixels)} for name, values in moments.items()}
        lanes_meta = [{"path": lane["path"], "manifest_sha256": lane["manifest_sha256"],
                       "days": lane["days"], "lane": lane["lane"],
                       "arrays": {key: array.record() for key, array in lane["arrays"].items()}} for lane in lanes]
        summary = {
            "schema": "direct450_merge_v1",
            "state": "completed",
            "completed_days": list(expected_days),
            "pixels": pixels,
            "nside": nside,
            "ordering": "NEST",
            "half_baselines": len(expected_days) * RECORDS_PER_DAY,
            "scope": "Direct used the frozen sky and orbit protocol. The historical STS run used the same documented sky and geometry configuration with independently generated FP32 visibilities. Historical consumed per-segment visibility hashes are unavailable.",
            "direct_target": "Corrected DCF weights, each method retains its own denominator. No output-domain exclusion.",
            "merge": "Sum lane N and W in FP64 by increasing first assigned day, then divide once and store FP32 C. Summation association is not a chronological day-by-day FP64 reduction.",
            "region_basis": "GPU native latitude tags" if args.regions else "Analytic HEALPix NEST native latitude",
            "affine_fit": "reference = alpha * candidate + beta; fitted on the full domain and unchanged in regions",
            "metric_limits": {"sts_vs_dcf_raw_n": "Different output units; raw error is descriptive only.",
                              "vs_input": "Dirty responses include sampling, masks and weighting; no deconvolution."},
            "protocol": lanes[0]["protocol"],
            "protocol_sha256": protocol_sha,
            "lanes_in_merge_order": lanes_meta,
            "sts_archive_manifest_sha256": archive_sha,
            "input_files": {key: array.record() for key, array in input_arrays.items()},
            "output_files": outputs,
            "direct_extrema": extrema,
            "comparisons": metrics,
        }
        with (staging / "merge_summary.json").open("x") as handle:
            json.dump(summary, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists():
            raise ValueError("Output directory appeared before final commit")
        os.rename(staging, target)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", nargs="+", required=True, type=Path, help="Lane roots, CURRENT.json or committed generation manifests")
    for key in ("input", "sts", "sts-w", "sts-manifest", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--input-sha", default=SKY_SHA256)
    parser.add_argument("--protocol-sha", help="Check an explicitly supplied run SHA256 against the pinned scientific protocol")
    parser.add_argument("--regions", type=Path, help="Optional aligned GPU native latitude uint8 tags")
    parser.add_argument("--regions-sha")
    parser.add_argument("--chunk", type=int, default=1 << 20)
    parser.add_argument("--pooled-only", action="store_true", help="Retain only C plus summaries to limit output size")
    args = parser.parse_args()
    if args.chunk <= 0 or bool(args.regions) != bool(args.regions_sha):
        parser.error("Chunk must be positive; --regions and --regions-sha must be provided together")
    for label in ("input", "regions", "protocol"):
        expected = getattr(args, label + "_sha")
        if expected:
            sha_text(expected, label)
    result = merge(args)
    for name, data in result["comparisons"].items():
        print(f"{name}: {json.dumps(data['regions']['all'], allow_nan=False)}")
    print(f"Final merge committed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
