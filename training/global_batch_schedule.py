from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BatchSlot:
    dataset_index: int
    valid: bool


class FixedGlobalBatchScheduler:
    """Stateful shuffled batches whose effective size is GPU-count invariant."""

    def __init__(self, dataset_size: int, global_batch_size: int, seed: int) -> None:
        if dataset_size < 1 or global_batch_size < 1:
            raise ValueError("dataset_size and global_batch_size must be positive")
        self.dataset_size = int(dataset_size)
        self.global_batch_size = int(global_batch_size)
        self.rng = np.random.default_rng(int(seed))
        self.order = self.rng.permutation(self.dataset_size).astype(np.int64)
        self.cursor = 0
        self.epoch = 0

    def next_batch(self) -> list[int]:
        result: list[int] = []
        while len(result) < self.global_batch_size:
            remaining = self.dataset_size - self.cursor
            take = min(self.global_batch_size - len(result), remaining)
            result.extend(self.order[self.cursor : self.cursor + take].tolist())
            self.cursor += take
            if self.cursor == self.dataset_size:
                self.epoch += 1
                self.order = self.rng.permutation(self.dataset_size).astype(np.int64)
                self.cursor = 0
        return result

    def state_dict(self) -> dict[str, object]:
        return {
            "dataset_size": self.dataset_size,
            "global_batch_size": self.global_batch_size,
            "order": self.order.copy(),
            "cursor": self.cursor,
            "epoch": self.epoch,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if int(state["dataset_size"]) != self.dataset_size:
            raise ValueError("Scheduler dataset size changed")
        if int(state["global_batch_size"]) != self.global_batch_size:
            raise ValueError("Scheduler global batch size changed")
        order = np.asarray(state["order"], dtype=np.int64)
        if order.shape != (self.dataset_size,) or not np.array_equal(
            np.sort(order), np.arange(self.dataset_size)
        ):
            raise ValueError("Invalid scheduler order")
        self.order = order.copy()
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        if not 0 <= self.cursor < self.dataset_size:
            raise ValueError("Invalid scheduler cursor")
        self.rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]


def slots_for_rank(
    batch: list[int], world_size: int, rank: int
) -> tuple[list[BatchSlot], float]:
    if not batch or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid batch or distributed rank")
    if len(batch) < world_size:
        raise ValueError("global batch size must be at least the number of ranks")
    # Uneven local counts are safe here: each rank has exactly one synchronized
    # backward (its last item), while earlier local items use DDP.no_sync(). This
    # avoids expensive zero-weight full-physics padding when 8 items use 3--6 GPUs.
    slots = [BatchSlot(dataset_index=batch[position], valid=True) for position in range(rank, len(batch), world_size)]
    # DDP averages over ranks. This scale produces the mean of the real global batch.
    return slots, world_size / len(batch)
