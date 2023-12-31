from config import USE_MLX
if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch
import torch.nn as torch_nn

from base_benchmark import BaseBenchmark
from utils import load_mnist


class Conv2d(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hid = kwargs["hid"]
        self.channels = kwargs["channels"]

    def setup(self, **kwargs):
        data = load_mnist().data / 255.0

        C = self.channels
        # Reshape in 4D array to apply conv2d.
        data_torch = data.view(-1, C, 28, 28)

        # On mlx, the channels are the last dimension.
        data_mlx = data.view(-1, 28, 28, C)

        return data_mlx.numpy(), data_torch.numpy()

    def forward_mlx(self, data, **kwargs):
        in_dim = data.shape[-1]
        layer = mx_nn.Conv2d(in_dim, self.hid, kernel_size=3, stride=3, padding=1)

        y = layer(data)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, data, **kwargs):
        in_dim = data.shape[1]
        layer = torch_nn.Conv2d(
            in_dim, self.hid, kernel_size=3, stride=3, padding=1, device=self.device
        )

        y = layer(data)
        self.sync_mps_if_needed()
