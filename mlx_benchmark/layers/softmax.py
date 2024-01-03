from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark


class Softmax(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, **kwargs):
        a = self.inputs[0]

        y = mx.softmax(a, axis=-1)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a = self.inputs[0]

        y = F.softmax(a, dim=-1)
        self.sync_mps_if_needed()
