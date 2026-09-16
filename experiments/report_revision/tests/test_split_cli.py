import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]


class SplitCliTests(unittest.TestCase):
    def test_cli_writes_frozen_manifest_with_layout_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cases = []
            for case_id in range(12):
                cases.append({
                    "id": case_id,
                    "blocks": [{
                        "x": (case_id // 2) * 10,
                        "y": 0,
                        "w": 10,
                        "h": 10,
                        "power_mw": 1 + case_id,
                    }],
                })
            params = root / "cases_params.json"
            params.write_text(json.dumps(cases), encoding="utf-8")
            output = root / "split.json"

            completed = subprocess.run([
                sys.executable, str(EXPERIMENT_ROOT / "freeze_splits.py"),
                "--params", str(params), "--train-sizes", "3,6",
                "--test-fraction", "0.25", "--seed", "9", "--replicates", "3",
                "--output", str(output),
            ], text=True, capture_output=True)

            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(len(manifest["split"]["train_subsets"]["3"]), 3)
            self.assertIn("source_sha256", manifest)
            self.assertEqual(set(manifest["replicates"]), {"0", "1", "2"})


if __name__ == "__main__":
    unittest.main()
