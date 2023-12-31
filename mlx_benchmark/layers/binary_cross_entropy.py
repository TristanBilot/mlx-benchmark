import torch
import mlx.core as mx
import mlx.nn as mx_nn
import torch.nn as torch_nn
import torch.nn.functional as F

from base_benchmark import BaseBenchmark
from utils import load_mnist


class BCE(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dim = kwargs["dim"].split("x")
        self.shape = [int(d) for d in dim]

    def setup(self, **kwargs):
        data = torch.rand(self.shape)

        return data.numpy(), data.numpy()

    def forward_mlx(self, data, **kwargs):
        y = mx.random.randint(shape=data.shape, low=0, high=2)

        y = mx_nn.losses.binary_cross_entropy(data, y)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, data, **kwargs):
        y = torch.randint(
            size=data.shape, dtype=torch.float32, low=0, high=2, device=self.device
        )

        y = F.binary_cross_entropy(data, y)
        self.sync_mps_if_needed()
