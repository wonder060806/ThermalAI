import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[2]


class ProvenanceTests(unittest.TestCase):
    def test_cli_records_upstream_commits_dirty_state_and_artifact_hashes(self):
        if not (WORKSPACE / "3d-ice" / ".git").is_dir() or not (WORKSPACE / "DeepOHeat" / ".git").is_dir():
            self.skipTest("upstream git checkouts unavailable")
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "provenance.json"
            completed = subprocess.run([
                sys.executable, str(ROOT / "capture_provenance.py"),
                "--workspace-root", str(WORKSPACE), "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(len(report["repositories"]["3d-ice"]["commit"]), 40)
            self.assertEqual(len(report["repositories"]["DeepOHeat"]["commit"]), 40)
            self.assertIn("dirty", report["repositories"]["DeepOHeat"])
            self.assertIn("sha256", report["artifacts"]["pinn_checkpoint"])
            self.assertIn("sha256", report["artifacts"]["3d_ice_emulator"])


if __name__ == "__main__":
    unittest.main()
