import torch
import mlx.core as mx
import mlx.nn as mx_nn
import torch.nn as torch_nn

from base_benchmark import BaseBenchmark
from utils import load_mnist


class MatMul(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dim = kwargs["dim"].split("x")
        self.shape = [int(d) for d in dim]

    def setup(self, **kwargs):
        data = torch.randn(self.shape)

        return data.numpy(), data.numpy()

    def forward_mlx(self, data, **kwargs):
        mat2 = (
            data.reshape(
                *data.shape[:-2], data.shape[-1], data.shape[-2]
            )  # with batch dim
            if data.ndim > 2
            else mx.transpose(data)
        )

        y = data @ mat2
        mx.eval(y)

    def forward_torch(self, data, **kwargs):
        mat2 = (
            data.reshape(
                *data.shape[:-2], data.shape[-1], data.shape[-2]
            )  # with batch dim
            if data.ndim > 2
            else data.T
        )

        y = data @ mat2
        self.sync_mps_if_needed()
