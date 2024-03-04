from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch

from base_benchmark import BaseBenchmark


class Concat(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, compile=False, **kwargs):
        a, b = self.inputs

        fn = self.compile_if_needed(mx.concatenate, compile=compile)
        y = fn([a, b], self.kwargs["axis"])
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.inputs

        y = torch.cat([a, b], dim=self.kwargs["axis"])
        self.sync_torch_gpu_if_needed()
