#!/usr/bin/env python3
"""CPU-only command-planning tests: never launch MPI jobs or query GPUs.

The fake executables are tripwires. Even an MPI --version or nvidia-smi call
would be recorded and fail these prepare-only checks.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/launch_background.py"


def following(command, token):
    return command[command.index(token) + 1]


class LauncherPlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.tripwire_log = self.root / "forbidden_invocations.jsonl"
        bindir = self.root / "fakebin"
        bindir.mkdir()
        fake = (f"#!{sys.executable}\n"
                "import json,os,sys\n"
                "with open(os.environ['PLANNING_TRIPWIRE_LOG'],'a') as f:\n"
                " f.write(json.dumps({'executable':sys.argv[0],'args':sys.argv[1:]})+'\\n')\n"
                "sys.exit(93)\n")
        for name in ("mpiexec", "mpirun", "nvidia-smi", "fake_driver"):
            p = bindir / name
            p.write_text(fake)
            p.chmod(0o755)
        self.fake_mpi = bindir / "mpiexec"
        self.fake_binary = bindir / "fake_driver"
        self.hostfile = self.root / "hosts.actual"
        self.hostfile.write_text("nodeA\nnodeB\nnodeC\n")
        self.env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ.get("PATH", ""),
                        PLANNING_TRIPWIRE_LOG=str(self.tripwire_log))
        self.env.pop("CUDA_VISIBLE_DEVICES", None)
        self.env.pop("MPIEXEC", None)
        self.counter = 0

    def tearDown(self):
        self.tmp.cleanup()

    def invoke(self, extra=(), flavor="mpich", expected_success=True, env_extra=None):
        self.counter += 1
        out = self.root / f"plan{self.counter}"
        command = [sys.executable, str(SCRIPT), "--mode=cluster", "--prepare-only",
                   f"--mpi-flavor={flavor}", f"--hostfile={self.hostfile}", f"--run-dir={out}",
                   f"--mpi-launcher={self.fake_mpi}", f"--binary={self.fake_binary}", *extra]
        result = subprocess.run(command, env={**self.env, **(env_extra or {})}, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        if expected_success:
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            manifest = json.loads((out / "launch_manifest.json").read_text())
            status = json.loads((out / "status.json").read_text())
            self.assertTrue(manifest["prepare_only"])
            self.assertEqual(manifest["state"], "planned")
            self.assertEqual(status["state"], "planned")
            self.assertFalse((out / ".queue.lock").exists())
            self.assertFalse((out / "launch_ack.json").exists())
            self.assertNotIn("pid", status)
            self.assertIsNone(manifest.get("one_time_memory_preflight"))
        else:
            self.assertNotEqual(result.returncode, 0, "invalid plan was accepted")
            manifest = None
        self.assertFalse(self.tripwire_log.exists(), "prepare-only invoked an MPI/GPU executable")
        return manifest, result

    def check_matrix(self, flavor, gpus_per_node, extra=()):
        manifest, _ = self.invoke(["--nodes=1,2,3", f"--gpus-per-node={gpus_per_node}", *extra], flavor)
        self.assertEqual(manifest["gpus_per_node"], gpus_per_node)
        self.assertEqual(manifest["environment"]["CUDA_VISIBLE_DEVICES"], ",".join(str(i) for i in range(gpus_per_node)))
        self.assertFalse(manifest["cross_node_results_claimed"])
        self.assertFalse(manifest["continuous_resource_monitoring"])
        steps = manifest["steps"]
        self.assertEqual(len(steps), 6)
        self.assertEqual([s["ranks"] for s in steps], [gpus_per_node, gpus_per_node,
                                                      2 * gpus_per_node, 2 * gpus_per_node,
                                                      3 * gpus_per_node, 3 * gpus_per_node])
        self.assertEqual([s["node_count"] for s in steps], [1, 1, 2, 2, 3, 3])
        for step in steps:
            command = step["command"]
            self.assertEqual(step["gpus_per_node"], gpus_per_node)
            self.assertEqual(step["ranks"], step["node_count"] * gpus_per_node)
            self.assertIn(f"nodes{step['node_count']}_gpn{gpus_per_node}_ranks{step['ranks']}", step["name"])
            self.assertEqual(int(following(command, "-np" if flavor == "openmpi" else "-n")), step["ranks"])
            if flavor == "openmpi":
                self.assertEqual(following(command, "--map-by"), f"ppr:{gpus_per_node}:node")
                self.assertEqual(following(command, "--bind-to"), "none")
                self.assertEqual(following(command, "--hostfile"), str(self.hostfile))
                self.assertNotIn("-ppn", command)
            else:
                self.assertEqual(following(command, "-ppn"), str(gpus_per_node))
                self.assertEqual(following(command, "-f"), str(self.hostfile))
                self.assertNotIn("--map-by", command)
            self.assertNotIn("--allow-shared-gpu=1", command)
        return manifest

    def test_two_gpu_per_node_openmpi_plan(self):
        self.check_matrix("openmpi", 2)

    def test_two_gpu_per_node_mpich_plan(self):
        self.check_matrix("mpich", 2)

    def test_one_gpu_per_node_plan_is_preserved(self):
        self.check_matrix("mpich", 1)

    def test_explicit_ranks_two_four_six_infer_one_two_three_nodes(self):
        manifest, _ = self.invoke(["--ranks=2,4,6", "--gpus-per-node=2"])
        self.assertEqual([s["ranks"] for s in manifest["steps"]], [2, 2, 4, 4, 6, 6])
        self.assertEqual([s["node_count"] for s in manifest["steps"]], [1, 1, 2, 2, 3, 3])

    def test_nodes_and_ranks_are_mutually_exclusive(self):
        self.invoke(["--nodes=1,2", "--ranks=2,4", "--gpus-per-node=2"], expected_success=False)

    def test_rank_count_not_divisible_by_gpu_per_node_rejected(self):
        self.invoke(["--ranks=3", "--gpus-per-node=2"], expected_success=False)

    def test_zero_gpus_per_node_rejected(self):
        self.invoke(["--nodes=1", "--gpus-per-node=0"], expected_success=False)

    def test_duplicate_node_counts_rejected(self):
        self.invoke(["--nodes=1,1", "--gpus-per-node=2"], expected_success=False)

    def test_insufficient_visible_gpu_list_rejected(self):
        self.invoke(["--nodes=1", "--gpus-per-node=2"], env_extra={"CUDA_VISIBLE_DEVICES": "0"}, expected_success=False)

    def test_duplicate_visible_gpu_list_rejected(self):
        self.invoke(["--nodes=1", "--gpus-per-node=2"], env_extra={"CUDA_VISIBLE_DEVICES": "0,0"}, expected_success=False)

    def test_two_strategies_expand_matrix_without_executing_it(self):
        manifest, _ = self.invoke(["--nodes=1,2,3", "--gpus-per-node=2", "--strategies=flat,hierarchical"])
        self.assertEqual(len(manifest["steps"]), 12)
        actual = set()
        for step in manifest["steps"]:
            strategy = step["strategy"]
            self.assertIn(strategy, ("flat", "hierarchical"))
            self.assertIn("--strategy=" + strategy, step["command"])
            mode_flags = [x for x in step["command"] if x.startswith("--recon-mode=")]
            self.assertEqual(len(mode_flags), 1)
            actual.add((step["node_count"], step["ranks"], strategy, mode_flags[0].split("=", 1)[1]))
        expected = {(nodes, 2 * nodes, strategy, mode)
                    for nodes in (1, 2, 3) for strategy in ("flat", "hierarchical") for mode in ("rep", "exact")}
        self.assertEqual(actual, expected)

    def test_duplicate_strategies_rejected(self):
        self.invoke(["--nodes=1", "--gpus-per-node=2", "--strategies=flat,flat"], expected_success=False)

    def test_unknown_strategy_rejected(self):
        self.invoke(["--nodes=1", "--gpus-per-node=2", "--strategies=unknown"], expected_success=False)


if __name__ == "__main__":
    unittest.main()
