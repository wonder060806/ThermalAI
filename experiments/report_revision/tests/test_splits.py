import sys
import unittest
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_ROOT))

from revision.splits import layout_group_id, make_nested_group_split, make_replicated_train_subsets


class NestedSplitTests(unittest.TestCase):
    def setUp(self):
        self.case_ids = [f"case_{i}" for i in range(12)]
        self.group_ids = [f"layout_{i // 2}" for i in range(12)]

    def test_split_is_deterministic_nested_and_group_isolated(self):
        first = make_nested_group_split(
            self.case_ids, self.group_ids, train_sizes=[3, 6], test_fraction=0.25, seed=9
        )
        second = make_nested_group_split(
            self.case_ids, self.group_ids, train_sizes=[3, 6], test_fraction=0.25, seed=9
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["train_subsets"]["3"]), 3)
        self.assertEqual(len(first["train_subsets"]["6"]), 6)
        self.assertTrue(
            set(first["train_subsets"]["3"]).issubset(first["train_subsets"]["6"])
        )
        self.assertFalse(
            set(first["train_subsets"]["6"]) & set(first["test"])
        )

        by_case = dict(zip(self.case_ids, self.group_ids))
        train_groups = {by_case[c] for c in first["train_pool"]}
        test_groups = {by_case[c] for c in first["test"]}
        self.assertFalse(train_groups & test_groups)

    def test_rejects_duplicate_case_ids(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            make_nested_group_split(
                ["a", "a"], ["g1", "g2"], [1], test_fraction=0.2, seed=1
            )

    def test_rejects_train_size_larger_than_available_pool(self):
        with self.assertRaisesRegex(ValueError, "largest train size"):
            make_nested_group_split(
                self.case_ids, self.group_ids, [11], test_fraction=0.25, seed=1
            )

    def test_layout_group_ignores_power_but_not_geometry(self):
        first = [
            {"x": 0, "y": 10, "w": 20, "h": 30, "power_mw": 1.0},
            {"x": 40, "y": 50, "w": 10, "h": 10, "power_mw": 2.0},
        ]
        reordered_new_power = [
            {"x": 40, "y": 50, "w": 10, "h": 10, "power_mw": 9.0},
            {"x": 0, "y": 10, "w": 20, "h": 30, "power_mw": 8.0},
        ]
        moved = [{"x": 1, "y": 10, "w": 20, "h": 30, "power_mw": 1.0}]

        self.assertEqual(layout_group_id(first), layout_group_id(reordered_new_power))
        self.assertNotEqual(layout_group_id(first), layout_group_id(moved))

    def test_replicates_share_test_set_but_vary_nested_training_cases(self):
        base = make_nested_group_split(
            [str(i) for i in range(30)], [f"g{i}" for i in range(30)],
            train_sizes=[5, 10], test_fraction=0.2, seed=9,
        )
        replicates = make_replicated_train_subsets(base, [5, 10], 3, seed=100)
        self.assertEqual(set(replicates), {"0", "1", "2"})
        selections = []
        for replicate in replicates.values():
            small = replicate["train_subsets"]["5"]
            large = replicate["train_subsets"]["10"]
            self.assertTrue(set(small).issubset(large))
            self.assertFalse(set(large) & set(base["test"]))
            selections.append(tuple(small))
        self.assertGreater(len(set(selections)), 1)


if __name__ == "__main__":
    unittest.main()
