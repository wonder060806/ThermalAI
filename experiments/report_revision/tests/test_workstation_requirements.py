import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class WorkstationRequirementsTests(unittest.TestCase):
    def test_includes_legacy_thermalai_benchmark_dependencies(self):
        requirements = (
            ROOT / "requirements-workstation.txt"
        ).read_text(encoding="utf-8").lower()
        for package in ("matplotlib", "ordered-set", "gstools", "smt"):
            with self.subTest(package=package):
                self.assertIn(package, requirements)


if __name__ == "__main__":
    unittest.main()
