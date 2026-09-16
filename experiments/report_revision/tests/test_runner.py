import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.runner import (
    assign_jobs,
    build_physics_weight_jobs,
    build_realistic_3d_jobs,
    build_small_sample_jobs,
    gpu_queues,
    latency_summary,
    pending_jobs,
    run_job,
)


class RunnerTests(unittest.TestCase):
    def test_realistic_3d_matrix_has_five_independent_seeds(self):
        jobs = build_realistic_3d_jobs(
            "python", "train_3d_revision.py", "split.json", "data",
            "results", train_size=200, epochs=1000, learning_rate=1e-4,
        )
        self.assertEqual(len(jobs), 5)
        self.assertEqual({job["init_seed"] for job in jobs}, set(range(5)))
        self.assertTrue(all(job["family"] == "realistic_3d" for job in jobs))

    def test_physics_matrix_has_paired_variants_weights_and_seeds(self):
        jobs = build_physics_weight_jobs(
            python_executable="python", train_script="train_revision.py",
            split_path="split.json", data_dir="data", checkpoint="pinn.pth",
            output_dir="physics", train_size=50, epochs=2000,
            learning_rate=1e-4,
        )
        self.assertEqual(len(jobs), 110)
        combinations = {(job["physics_variant"], job["physics_weight"], job["init_seed"]) for job in jobs}
        for seed in range(10):
            self.assertIn(("none", 0.0, seed), combinations)
            for variant in ("surface_flux", "volumetric_source"):
                for weight in (0.01, 0.05, 0.1, 0.25, 0.5):
                    self.assertIn((variant, weight, seed), combinations)

    def test_latency_summary_reports_median_and_nearest_rank_p95(self):
        result = latency_summary([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(result["count"], 5)
        self.assertEqual(result["median_ms"], 3.0)
        self.assertEqual(result["p95_ms"], 100.0)

    def test_assign_jobs_respects_preferred_gpu_and_is_deterministic(self):
        jobs = [
            {"job_id": "pinn-1", "family": "small_sample_pinn"},
            {"job_id": "random-1", "family": "small_sample_random"},
            {"job_id": "physics-1", "family": "physics_loss"},
            {"job_id": "real-1", "family": "realistic_3d"},
            {"job_id": "pinn-2", "family": "small_sample_pinn"},
        ]
        assigned = assign_jobs(jobs, gpu_ids=[0, 1, 2, 3])
        self.assertEqual([job["gpu_id"] for job in assigned], [0, 1, 2, 3, 0])
        self.assertEqual(assigned, assign_jobs(jobs, gpu_ids=[0, 1, 2, 3]))

    def test_pending_jobs_skips_only_matching_success_marker(self):
        jobs = [
            {"job_id": "a", "config_hash": "hash-a"},
            {"job_id": "b", "config_hash": "hash-b"},
        ]
        markers = {
            "a": {"status": "completed", "config_hash": "hash-a"},
            "b": {"status": "completed", "config_hash": "old-hash"},
        }
        self.assertEqual([job["job_id"] for job in pending_jobs(jobs, markers)], ["b"])

    def test_gpu_queues_preserve_order_per_device(self):
        queues = gpu_queues([
            {"job_id": "a", "gpu_id": 0},
            {"job_id": "b", "gpu_id": 1},
            {"job_id": "c", "gpu_id": 0},
        ])
        self.assertEqual([job["job_id"] for job in queues[0]], ["a", "c"])
        self.assertEqual([job["job_id"] for job in queues[1]], ["b"])

    def test_small_sample_matrix_is_paired_and_has_seventy_jobs(self):
        jobs = build_small_sample_jobs(
            python_executable="python",
            train_script="train_revision.py",
            split_path="formal_split.json",
            data_dir="data",
            checkpoint="pinn.pth",
            output_dir="results",
            epochs=2000,
            learning_rate=1e-4,
            power_per_unit_mw=250.0,
        )
        self.assertEqual(len(jobs), 70)
        pairs = {(job["train_size"], job["init_seed"], job["method"]) for job in jobs}
        for size in (10, 20, 50, 100, 200):
            expected_seeds = range(10) if size in (10, 20) else range(5)
            for seed in expected_seeds:
                self.assertIn((size, seed, "pinn"), pairs)
                self.assertIn((size, seed, "random"), pairs)
        budgets = {(job["epochs"], job["learning_rate"]) for job in jobs}
        self.assertEqual(budgets, {(2000, 1e-4)})
        self.assertTrue(all("--power-per-unit-mw" in job["command"] for job in jobs))
        self.assertTrue(all("--data-seed" in job["command"] for job in jobs))
        self.assertTrue(all(job["data_seed"] == job["init_seed"] for job in jobs))
        assigned = assign_jobs(jobs, [0, 1, 2, 3])
        counts = [sum(job["gpu_id"] == gpu for job in assigned) for gpu in range(4)]
        self.assertEqual(counts, [18, 18, 17, 17])

    def test_dry_run_cli_emits_all_jobs_without_launching_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"jobs": [
                {"job_id": "one", "family": "small_sample_pinn", "command": ["never-run"]},
                {"job_id": "two", "family": "physics_loss", "command": ["never-run"]},
            ]}), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(EXPERIMENT_ROOT / "workstation_runner.py"),
                 "--matrix", str(matrix), "--dry-run"],
                text=True, capture_output=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            output = json.loads(completed.stdout)
            self.assertEqual(output["job_count"], 2)
        self.assertEqual([job["gpu_id"] for job in output["jobs"]], [0, 1])

    def test_run_job_sets_gpu_and_writes_completion_marker_and_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            job = {
                "job_id": "smoke",
                "config_hash": "abc123",
                "gpu_id": 3,
                "command": [
                    sys.executable,
                    "-c",
                    "import os; print('gpu=' + os.environ['CUDA_VISIBLE_DEVICES'])",
                ],
            }
            marker = run_job(job, root)

            self.assertEqual(marker["status"], "completed")
            self.assertEqual(marker["exit_code"], 0)
            self.assertIn("gpu=3", (root / "smoke.log").read_text(encoding="utf-8"))
            stored = json.loads((root / "smoke.marker.json").read_text(encoding="utf-8"))
            self.assertEqual(stored["config_hash"], "abc123")

    def test_runner_cli_execution_writes_summary_and_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"jobs": [{
                "job_id": "live-smoke",
                "family": "small_sample_random",
                "command": [sys.executable, "-c", "print('trained')"],
            }]}), encoding="utf-8")
            artifacts = root / "artifacts"
            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "workstation_runner.py"),
                "--matrix", str(matrix), "--artifacts", str(artifacts),
            ], text=True, capture_output=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["completed"], 1)
            self.assertTrue((artifacts / "live-smoke.marker.json").exists())


if __name__ == "__main__":
    unittest.main()
