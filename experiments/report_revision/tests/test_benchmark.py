import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.benchmarking import measure_callable, measure_command


class BenchmarkingTests(unittest.TestCase):
    def test_callable_runs_warmups_outside_recorded_samples(self):
        calls = []
        result = measure_callable(lambda: calls.append(1), warmups=2, repeats=3)
        self.assertEqual(len(calls), 5)
        self.assertEqual(result["summary"]["count"], 3)
        self.assertEqual(len(result["samples_ms"]), 3)

    def test_command_measurement_includes_process_start_and_checks_exit(self):
        result = measure_command(
            [sys.executable, "-c", "print('ok')"], repeats=2
        )
        self.assertEqual(result["summary"]["count"], 2)
        with self.assertRaisesRegex(RuntimeError, "exit code"):
            measure_command([sys.executable, "-c", "raise SystemExit(4)"], repeats=1)


if __name__ == "__main__":
    unittest.main()
