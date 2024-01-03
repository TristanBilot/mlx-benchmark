from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch

from base_benchmark import BaseBenchmark


class MatMul(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, **kwargs):
        a, b = self.inputs

        y = a @ b
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.inputs

        y = a @ b
        self.sync_mps_if_needed()
