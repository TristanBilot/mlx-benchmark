from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch

from base_benchmark import BaseBenchmark


class MatMul(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, compile=False, **kwargs):
        a, b = self.inputs

        fn = lambda x, y: x @ y
        fn = self.compile_if_needed(fn, compile=compile)

        y = fn(a, b)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.inputs

        y = a @ b
        self.sync_torch_gpu_if_needed()
