from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark


class Linear(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def additional_preprocessing(self, framework, device):
        _, b, _ = self.inputs

        if framework == "torch":
            self.b_torch = b.T

    def forward_mlx(self, **kwargs):
        a, b, c = self.inputs

        y = a @ b + c
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, _, c = self.inputs
        b = self.b_torch

        y = F.linear(a, b, c)
        self.sync_mps_if_needed()
