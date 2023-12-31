import torch
import torch.nn.functional as F
import mlx.core as mx

from base_benchmark import BaseBenchmark


class Softmax(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dim = kwargs["dim"].split("x")
        self.shape = [int(d) for d in dim]

    def setup(self, **kwargs):
        data = torch.randn(self.shape)

        return data.numpy(), data.numpy()

    def forward_mlx(self, data, **kwargs):
        y = mx.softmax(data, axis=-1)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, data, **kwargs):
        y = F.softmax(data, dim=-1)
        self.sync_mps_if_needed()
