from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark


class Conv1d(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def additional_preprocessing(self, framework, device):
        a, b = self.inputs

        # In torch, the channels are located at axis 1
        if framework == "torch":
            self.a_torch = torch.transpose(a, -1, -2)
            self.b_torch = torch.transpose(b, -1, -2)

    def forward_mlx(self, compile=False, **kwargs):
        a, b = self.inputs

        fn = self.compile_if_needed(mx.conv1d, compile=compile)
        y = fn(a, b)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.a_torch, self.b_torch

        y = F.conv1d(a, b)
        self.sync_torch_gpu_if_needed()


class Conv2d(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def additional_preprocessing(self, framework, device):
        a, b = self.inputs

        if framework == "torch":
            self.a_torch = torch.permute(a, (0, 3, 1, 2))
            self.b_torch = torch.permute(b, (0, 3, 1, 2))

    def forward_mlx(self, compile=False, **kwargs):
        a, b = self.inputs

        fn = self.compile_if_needed(mx.conv2d, compile=compile)
        y = fn(a, b)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.a_torch, self.b_torch

        y = F.conv2d(a, b)
        self.sync_torch_gpu_if_needed()
