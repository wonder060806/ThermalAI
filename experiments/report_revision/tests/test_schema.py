import json
import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.schema import build_manifest, canonical_json, config_hash


class SchemaTests(unittest.TestCase):
    def test_canonical_json_and_hash_ignore_mapping_key_order(self):
        first = {"train": {"epochs": 20, "seed": 7}, "sizes": [10, 20]}
        second = {"sizes": [10, 20], "train": {"seed": 7, "epochs": 20}}

        self.assertEqual(
            canonical_json(first),
            '{"sizes":[10,20],"train":{"epochs":20,"seed":7}}',
        )
        self.assertEqual(config_hash(first), config_hash(second))
        self.assertEqual(len(config_hash(first)), 64)

    def test_manifest_contains_traceability_fields_and_is_json_serializable(self):
        manifest = build_manifest(
            config={"experiment": "small_sample", "seed": 123},
            splits={"train": ["case_1"], "test": ["case_2"]},
            runtime={"hostname": "worker-a", "gpu": "RTX test"},
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["experiment_id"], config_hash(manifest["config"]))
        self.assertEqual(manifest["splits"]["test"], ["case_2"])
        self.assertEqual(manifest["runtime"]["hostname"], "worker-a")
        self.assertIn("created_at_utc", manifest)
        json.dumps(manifest)

    def test_manifest_rejects_overlapping_train_and_test_cases(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_manifest(
                config={"experiment": "bad_split"},
                splits={"train": ["case_1"], "test": ["case_1"]},
                runtime={"hostname": "worker-a"},
            )


if __name__ == "__main__":
    unittest.main()

