import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ssh_workstation_task.sh"


def shell_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt":
        drive, remainder = resolved[0].lower(), resolved[2:].replace("\\", "/")
        return f"/mnt/{drive}{remainder}"
    return resolved


def bash_prefix() -> list[str]:
    return ["wsl", "bash"] if os.name == "nt" else ["bash"]


@unittest.skipUnless(
    shutil.which("wsl") if os.name == "nt" else shutil.which("bash"),
    "A Bash runtime is required",
)
class SshWorkstationTaskTests(unittest.TestCase):
    def test_default_workstation_root_is_home_relative(self):
        script_text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn(
            "${HOME}/aicooling/ThermalAI-full/experiments/report_revision",
            script_text,
        )
        self.assertNotIn("/home/so", script_text)

    def _fixture(self, root: Path, stage: str = "small") -> Path:
        (root / "workstation_runner.py").write_text("# fixture", encoding="utf-8")
        (root / "configs").mkdir()
        matrix_names = {
            "small": "small_sample_workstation.json",
            "physics": "physics_workstation.json",
            "3d": "realistic_3d_workstation.json",
        }
        matrix = root / "configs" / matrix_names[stage]
        matrix.write_text("{}", encoding="utf-8")
        return matrix

    def test_bash_script_has_valid_syntax(self):
        completed = subprocess.run(
            [*bash_prefix(), "-n", shell_path(SCRIPT)], capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_render_small_contains_linux_runner_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            matrix = self._fixture(root)
            completed = subprocess.run([
                *bash_prefix(), shell_path(SCRIPT), "render", "small",
                shell_path(root), "/usr/bin/python3",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rendered = json.loads(completed.stdout)
            self.assertEqual(rendered["session_name"], "thermalai-small")
            self.assertEqual(rendered["python"], "/usr/bin/python3")
            self.assertEqual(rendered["matrix"], shell_path(matrix))
            self.assertIn("workstation_runner.py", rendered["runner_command"])
            self.assertIn("--gpus 0,1,2,3", rendered["runner_command"])
            self.assertIn("workstation_artifacts", rendered["runner_command"])
            self.assertTrue(rendered["log"].endswith("ssh_small.log"))

    def test_render_rejects_missing_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._fixture(root)
            completed = subprocess.run([
                *bash_prefix(), shell_path(SCRIPT), "render", "3d",
                shell_path(root), "/usr/bin/python3",
            ], capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("matrix does not exist", completed.stderr)


if __name__ == "__main__":
    unittest.main()
