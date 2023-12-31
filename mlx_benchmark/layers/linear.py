from config import USE_MLX
if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch
import torch.nn as torch_nn

from base_benchmark import BaseBenchmark
from utils import load_mnist


class Linear(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hid = kwargs["hid"]

    def setup(self, **kwargs):
        data = load_mnist().data / 255.0
        data = data.numpy()

        return data, data

    def forward_mlx(self, data, **kwargs):
        in_dim = data.shape[-1]
        layer = mx_nn.Linear(in_dim, self.hid)
        mx.eval(layer(data))

    @torch.no_grad()
    def forward_torch(self, data, **kwargs):
        in_dim = data.shape[-1]
        layer = torch_nn.Linear(in_dim, self.hid).to(self.device)
        layer(data)
        self.sync_mps_if_needed()
