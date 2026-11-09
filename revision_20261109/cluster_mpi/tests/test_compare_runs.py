#!/usr/bin/env python3
"""CPU-only negative-control tests for the output verification infrastructure."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_runs import compare, VerificationError


class ComparatorNegativeControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runs = [self.make_run(n) for n in (1, 2, 3)]

    def tearDown(self):
        self.temporary.cleanup()

    def make_run(self, ranks):
        path = self.root / f"np{ranks}"
        path.mkdir()
        config = {"nside": 8, "npix": 768, "tile_pix": 256, "recon_mode": "rep", "segs": 2,
                  "time_start": 0, "time_count": 8, "grid_cell_scale": 1, "orbit_seed": 42,
                  "day_start": 1, "day_count": 1, "tlen_limit": 0, "segments_limit": 0,
                  "btag": "10M", "lambda_m": 30, "sky_kind": "fixture", "input": "fixture.bin", "synthetic": False}
        meta = {"status": "completed", "config": config, "ranks": ranks,
                "input_fingerprint_fnv1a64": "known_fixture_fingerprint",
                "serial_check": {key: True for key in ("requested", "passed", "visibility_bitwise", "numerator_bitwise", "weight_bitwise", "map_bitwise")},
                "days": [{"day": 1, "dir": "day0001"}]}
        (path / "metrics.json").write_text(json.dumps(meta))
        numerator = (np.arange(768, dtype=np.float32) + 0.25).astype("<f4")
        weight = (1 + np.arange(768) % 3).astype("<u4")
        numerator[0] = 0
        weight[0] = 0
        output = np.zeros(768, dtype="<f4")
        output[1:] = numerator[1:] / weight[1:].astype(np.float32)
        arrays = {"numerator_f32.bin": numerator, "weight_u32.bin": weight, "map_f32.bin": output}
        day = path / "day0001"
        day.mkdir()
        for name, array in arrays.items():
            array.tofile(day / name)
        cursor = 0
        tile_cursor = 0
        lines = ["rank,pixel_start,pixel_count,tile_start,tile_count"]
        for rank in range(ranks):
            tiles = 3 // ranks + (rank < 3 % ranks)
            count = tiles * 256
            lines.append(f"{rank},{cursor},{count},{tile_cursor},{tiles}")
            shard = day / f"rank{rank:04d}"
            shard.mkdir()
            for name, array in arrays.items():
                array[cursor:cursor + count].tofile(shard / name)
            cursor += count
            tile_cursor += tiles
        (path / "ownership.csv").write_text("\n".join(lines) + "\n")
        return path

    def test_valid_unequal_two_rank_fixture(self):
        report = compare(self.runs)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["runs"][1]["ownership"][0]["pixel_count"], 512)
        self.assertEqual(report["runs"][1]["ownership"][1]["pixel_count"], 256)

    def test_corrupted_shard_is_not_hidden_by_good_global_map(self):
        p = self.runs[1] / "day0001/rank0001/map_f32.bin"
        x = np.fromfile(p, dtype="<f4")
        x[0] *= 2
        x.tofile(p)
        with self.assertRaisesRegex(VerificationError, "gather/order"):
            compare(self.runs)

    def test_consistently_averaging_normalized_shards_is_rejected(self):
        for p in (self.runs[1] / "day0001").rglob("map_f32.bin"):
            x = np.fromfile(p, dtype="<f4") / np.float32(2)
            x.tofile(p)
        with self.assertRaisesRegex(VerificationError, "numerator/weight"):
            compare(self.runs)

    def test_ownership_overlap_is_rejected(self):
        p = self.runs[1] / "ownership.csv"
        p.write_text(p.read_text().replace("1,512,256,2,1", "1,256,256,2,1"))
        with self.assertRaisesRegex(VerificationError, "overlap|gap"):
            compare(self.runs)

    def test_different_input_is_rejected(self):
        p = self.runs[2] / "metrics.json"
        x = json.loads(p.read_text())
        x["input_fingerprint_fnv1a64"] = "different_input"
        p.write_text(json.dumps(x))
        with self.assertRaisesRegex(VerificationError, "input fingerprints"):
            compare(self.runs)

    def test_completed_label_does_not_replace_missing_array(self):
        (self.runs[0] / "day0001/weight_u32.bin").unlink()
        with self.assertRaisesRegex(VerificationError, "missing array"):
            compare(self.runs)

    def test_failed_serial_gate_is_rejected(self):
        p = self.runs[0] / "metrics.json"
        x = json.loads(p.read_text())
        x["serial_check"]["visibility_bitwise"] = False
        p.write_text(json.dumps(x))
        with self.assertRaisesRegex(VerificationError, "visibility_bitwise"):
            compare(self.runs)

    def test_mixed_reduction_strategies_need_explicit_authorization(self):
        p = self.runs[2] / "metrics.json"
        x = json.loads(p.read_text())
        x["config"]["strategy"] = "hierarchical"
        p.write_text(json.dumps(x))
        with self.assertRaisesRegex(VerificationError, "different reduction strategies"):
            compare(self.runs)

    def test_explicit_cross_strategy_comparison_is_marked(self):
        p = self.runs[2] / "metrics.json"
        x = json.loads(p.read_text())
        x["config"]["strategy"] = "hierarchical"
        p.write_text(json.dumps(x))
        report = compare(self.runs, allow_strategy_difference=True)
        self.assertTrue(report["cross_strategy_comparison_explicitly_authorized"])
        self.assertEqual(report["runs"][2]["strategy"], "hierarchical")


if __name__ == "__main__":
    unittest.main()
