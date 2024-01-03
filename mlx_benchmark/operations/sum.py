from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch

from base_benchmark import BaseBenchmark


class Sum(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, **kwargs):
        a = self.inputs[0]

        y = mx.sum(a, axis=self.kwargs["axis"])
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a = self.inputs[0]

        y = torch.sum(a, dim=self.kwargs["axis"])
        self.sync_mps_if_needed()


class SumAll(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, **kwargs):
        a = self.inputs[0]

        y = mx.sum(a, axis=None)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a = self.inputs[0]

        y = torch.sum(a, dim=None)
        self.sync_mps_if_needed()
