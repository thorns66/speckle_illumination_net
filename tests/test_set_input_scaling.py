import unittest

import torch
from torch import nn

from models.set_encoder import SetEncoder


class _InputRecorder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []

    def forward(self, frames: torch.Tensor):
        self.seen.append(frames.detach().clone())
        return frames, frames[..., ::2, ::2], frames[..., ::4, ::4]


class SetInputScalingTest(unittest.TestCase):
    def test_shared_cnn_receives_unscaled_residual_frames(self):
        encoder = SetEncoder((1, 1, 1), frame_chunk_size=2, use_checkpoint=False)
        recorder = _InputRecorder()
        encoder.shared = recorder
        residual_frames = torch.tensor(
            [[[[[1.0, -2.0, 3.0, -4.0], [2.0, -1.0, 4.0, -3.0],
                [3.0, -4.0, 1.0, -2.0], [4.0, -3.0, 2.0, -1.0]]],
              [[[2.0, -4.0, 6.0, -8.0], [4.0, -2.0, 8.0, -6.0],
                [6.0, -8.0, 2.0, -4.0], [8.0, -6.0, 4.0, -2.0]]]]]
        )

        encoder.aggregate_frames(residual_frames)

        self.assertEqual(len(recorder.seen), 1)
        torch.testing.assert_close(
            recorder.seen[0],
            residual_frames.reshape(2, 1, 4, 4),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
