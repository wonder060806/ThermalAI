import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkstationPreflightCliTests(unittest.TestCase):
    def test_local_mode_writes_readiness_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            data.mkdir()
            (data / "cases_params.json").write_text(json.dumps([{"id": 0}, {"id": 1}]), encoding="utf-8")
            for case_id in (0, 1):
                (data / f"case_{case_id}.flp").write_text("x", encoding="utf-8")
                (data / f"case_{case_id}_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            script = root / "train.py"
            split = root / "split.json"
            checkpoint = root / "model.pth"
            for path in (script, checkpoint):
                path.write_text("x", encoding="utf-8")
            split.write_text(json.dumps({
                "split": {"test": ["1"], "train_subsets": {"1": ["0"]}}
            }), encoding="utf-8")
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"jobs": [{
                "job_id": "one", "command": [sys.executable, str(script),
                "--split", str(split), "--data-dir", str(data),
                "--checkpoint", str(checkpoint), "--output", str(root / "out.json")]
            }]}), encoding="utf-8")
            output = root / "preflight.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "workstation_preflight.py"),
                "--data-dir", str(data), "--matrix", str(matrix),
                "--minimum-cases", "1", "--expected-gpus", "0",
                "--train-sizes", "1",
                "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["dataset"]["case_count"], 2)

    def test_preflight_rejects_split_case_missing_from_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            data.mkdir()
            (data / "cases_params.json").write_text(json.dumps([{"id": 0}]), encoding="utf-8")
            (data / "case_0.flp").write_text("x", encoding="utf-8")
            (data / "case_0_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            script = root / "train.py"
            script.write_text("", encoding="utf-8")
            split = root / "split.json"
            split.write_text(json.dumps({"split": {"test": ["99"], "train_subsets": {"1": ["0"]}}}), encoding="utf-8")
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"jobs": [{"job_id": "one", "command": [
                sys.executable, str(script), "--split", str(split), "--data-dir", str(data),
                "--output", str(root / "out.json")]}]}), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(ROOT / "workstation_preflight.py"), "--data-dir", str(data),
                "--matrix", str(matrix), "--minimum-cases", "1", "--expected-gpus", "0",
                "--train-sizes", "1", "--output", str(root / "preflight.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("split references case IDs missing from dataset", completed.stderr)

    def test_preflight_checks_each_jobs_data_replicate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data = root / "data"
            data.mkdir()
            (data / "cases_params.json").write_text(json.dumps([{"id": 0}, {"id": 1}]), encoding="utf-8")
            for case_id in (0, 1):
                (data / f"case_{case_id}.flp").write_text("x", encoding="utf-8")
                (data / f"case_{case_id}_temp.txt").write_text("300 301\n302 303\n", encoding="utf-8")
            script = root / "train.py"
            script.write_text("", encoding="utf-8")
            split = root / "split.json"
            split.write_text(json.dumps({"split": {"test": ["1"], "train_subsets": {"1": ["0"]}},
                "replicates": {"0": {"train_subsets": {"1": ["0"]}}}}), encoding="utf-8")
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"jobs": [{"job_id": "bad-replicate", "command": [
                sys.executable, str(script), "--split", str(split), "--data-dir", str(data),
                "--train-size", "1", "--data-seed", "9", "--output", str(root / "out.json")]}]}), encoding="utf-8")
            completed = subprocess.run([
                sys.executable, str(ROOT / "workstation_preflight.py"), "--data-dir", str(data),
                "--matrix", str(matrix), "--minimum-cases", "1", "--expected-gpus", "0",
                "--output", str(root / "preflight.json"),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("replicate 9", completed.stderr)


if __name__ == "__main__":
    unittest.main()
