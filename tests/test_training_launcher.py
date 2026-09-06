import unittest

from train_dataset import _select_gpus


class TrainingLauncherTest(unittest.TestCase):
    def setUp(self):
        self.inventory = [
            {"index": index, "uuid": f"GPU-{index}", "name": "A40"}
            for index in range(6)
        ]

    def test_preserves_user_physical_gpu_order(self):
        selected = _select_gpus("5,1,3", self.inventory)
        self.assertEqual([item["index"] for item in selected], [5, 1, 3])

    def test_all_uses_inventory_order(self):
        selected = _select_gpus("all", self.inventory)
        self.assertEqual([item["index"] for item in selected], list(range(6)))

    def test_duplicate_and_unknown_gpus_are_rejected(self):
        with self.assertRaises(ValueError):
            _select_gpus("1,1", self.inventory)
        with self.assertRaises(ValueError):
            _select_gpus("7", self.inventory)


if __name__ == "__main__":
    unittest.main()
