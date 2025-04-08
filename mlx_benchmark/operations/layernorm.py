from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as nn

import torch
import torch.nn.functional as F
from base_benchmark import BaseBenchmark


class LayerNorm(BaseBenchmark):
    def additional_preprocessing(self, framework=None, device=None):
        if framework == "mlx":
            (x,) = self.inputs

            self.mlx_layernorm = nn.LayerNorm(
                dims=x.shape[2],
                affine=False,
                bias=False,
            )

        return

    def forward_mlx(self, compile=False, **kwargs):
        (x,) = self.inputs

        fn = self.mlx_layernorm
        fn = self.compile_if_needed(fn, compile=compile)

        y = fn(x)
        mx.eval(y)

    @torch.inference_mode()
    def forward_torch(self, **kwargs):
        (x,) = self.inputs

        fn = lambda x: F.layer_norm(
            x, normalized_shape=x.shape[2:], bias=None, weight=None
        )
        y = fn(x)

        self.sync_torch_gpu_if_needed()
