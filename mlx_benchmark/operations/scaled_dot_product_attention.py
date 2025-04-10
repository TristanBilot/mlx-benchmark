from config import USE_MLX
import math

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.core.fast

import torch
import torch.nn.functional as F 

from base_benchmark import BaseBenchmark


class ScaledDotProductAttention(BaseBenchmark):
    def additional_preprocessing(self, framework=None, device=None):
        q, k, v = self.inputs
        
        # verify input shapes
        batch_size, _, num_tokens_q, dim_q = q.shape
        batch_size_k, num_heads_k, num_tokens_k, dim_k = k.shape
        batch_size_v, num_heads_v, num_tokens_v, _ = v.shape
        
        if (batch_size_k != batch_size) or (batch_size_v != batch_size) or \
                (num_tokens_k != num_tokens_v) or (num_heads_k != num_heads_v) or (dim_q != dim_k):
            raise ValueError(
                f"incompatible shapes: q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}"
            )

        # create additive causal mask, scale
        attention_mask = torch.ones(num_tokens_q, num_tokens_k, dtype=torch.bool).tril(diagonal=0)
        attention_mask.masked_fill_(attention_mask.logical_not(), float("-inf"))

        if framework == "mlx":
            self.attention_mask_mlx = mx.array(attention_mask, dtype=q.dtype)

        if framework == "torch" and device is not None:
            self.attention_mask_torch = attention_mask.to(dtype=q.dtype, device=device)
            
        self.scale_factor = 1 / math.sqrt(dim_q)

        return

    def forward_mlx(self, compile=False, **kwargs):
        q, k, v = self.inputs

        fn = mlx.core.fast.scaled_dot_product_attention
        fn = self.compile_if_needed(fn, compile=compile)

        y = fn(q, k, v, scale=self.scale_factor, mask=self.attention_mask_mlx)
        mx.eval(y)

    @torch.inference_mode()
    def forward_torch(self, **kwargs):
        q, k, v = self.inputs

        fn = F.scaled_dot_product_attention

        y = fn(q, k, v, scale=self.scale_factor, attn_mask=self.attention_mask_torch)
        self.sync_torch_gpu_if_needed()
