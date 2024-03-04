from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark


class BCE(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def additional_preprocessing(self, framework, device):
        if framework == "torch":
            self.a_torch = torch.rand(*self.inputs[0].shape).to(device)
            self.b_torch = torch.randint(
                size=self.inputs[1].shape,
                dtype=torch.float32,
                low=0,
                high=2,
                device=device,
            )
        else:
            self.b_mlx = mx.random.randint(shape=self.inputs[1].shape, low=0, high=2)

    def forward_mlx(self, compile=False, **kwargs):
        a, _ = self.inputs
        b = self.b_mlx

        fn = self.compile_if_needed(mx_nn.losses.binary_cross_entropy, compile=compile)
        y = fn(a, b)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.a_torch, self.b_torch

        y = F.binary_cross_entropy(a, b)
        self.sync_torch_gpu_if_needed()
