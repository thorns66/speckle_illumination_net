import unittest

import numpy as np

from training.global_batch_schedule import FixedGlobalBatchScheduler, slots_for_rank


class GlobalBatchScheduleTest(unittest.TestCase):
    def test_one_to_six_ranks_keep_exact_global_mean_weight(self):
        batch = list(range(8))
        for world_size in range(1, 7):
            weights = np.zeros(len(batch), dtype=np.float64)
            valid_count = 0
            for rank in range(world_size):
                slots, backward_scale = slots_for_rank(batch, world_size, rank)
                for slot in slots:
                    if slot.valid:
                        weights[slot.dataset_index] += backward_scale / world_size
                        valid_count += 1
            self.assertEqual(valid_count, 8)
            np.testing.assert_allclose(weights, np.full(8, 1.0 / 8.0))

    def test_scheduler_resume_is_bit_exact(self):
        first = FixedGlobalBatchScheduler(80, 8, 20260901)
        prefix = [first.next_batch() for _ in range(7)]
        state = first.state_dict()
        expected = [first.next_batch() for _ in range(12)]

        resumed = FixedGlobalBatchScheduler(80, 8, 20260901)
        resumed.load_state_dict(state)
        actual = [resumed.next_batch() for _ in range(12)]
        self.assertEqual(expected, actual)
        self.assertEqual(len(prefix), 7)


if __name__ == "__main__":
    unittest.main()
