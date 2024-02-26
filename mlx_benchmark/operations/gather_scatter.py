from typing import Literal
from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch

from base_benchmark import BaseBenchmark
from utils import get_dummy_edge_index


class Gather(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def additional_preprocessing(self, framework, device):
        node_features, edge_index = self.inputs

        num_nodes = node_features.shape[0]
        edge_index = get_dummy_edge_index(
            (2, edge_index.shape[0]), num_nodes, device, framework
        )

        self.node_features = node_features
        self.index = edge_index[0]

    def forward_mlx(self, compile=False, **kwargs):
        a, b = self.node_features, self.index

        fn = lambda x, y: x[y]
        fn = self.compile_if_needed(fn, compile=compile)

        y = fn(a, b)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.node_features, self.index

        y = a[b]
        self.sync_torch_gpu_if_needed()


class _Scatter(BaseBenchmark):
    def __init__(self, scatter_op: Literal["indexing", "add", "max"], **kwargs):
        super().__init__(**kwargs)

        self.scatter_op = scatter_op

    def additional_preprocessing(self, framework, device):
        node_features, edge_index = self.inputs

        num_nodes = node_features.shape[0]
        edge_index = get_dummy_edge_index(
            (2, edge_index.shape[0]), num_nodes, device, framework
        )

        self.node_features = node_features[edge_index[0]]
        self.index = edge_index[1]
        self.src = (
            mx.zeros_like(node_features)
            if framework == "mlx"
            else torch.zeros_like(node_features)
        )

        if self.scatter_op != "indexing" and framework == "torch":
            self.index = self.index.unsqueeze(1)

        def fn_indexing(x, y, z):
            x[y] = z
            return x

        def fn_add(x, y, z):
            x.at[y].add(z)
            return x

        def fn_max(x, y, z):
            x.at[y].maximum(z)
            return x

        fns = {
            "indexing": fn_indexing,
            "add": fn_add,
            "max": fn_max,
        }
        self._scatter_fn_mlx = fns[self.scatter_op]

        def fn_indexing_torch(x, y, z):
            x[y] = z
            return x

        def fn_add_torch(x, y, z):
            x = torch.scatter_reduce(x, 0, y, z, reduce="sum")
            return x

        def fn_max_torch(x, y, z):
            x = torch.scatter_reduce(x, 0, y, z, reduce="max")
            return x

        fns = {
            "indexing": fn_indexing_torch,
            "add": fn_add_torch,
            "max": fn_max_torch,
        }
        self._scatter_fn_torch = fns[self.scatter_op]

    def forward_mlx(self, compile=False, **kwargs):
        a, b, c = self.src, self.index, self.node_features

        fn = self.compile_if_needed(self._scatter_fn_mlx, compile=compile)

        y = fn(a, b, c)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b, c = self.src, self.index, self.node_features

        y = self._scatter_fn_torch(a, b, c)
        self.sync_torch_gpu_if_needed()


class Scatter(_Scatter):
    def __init__(self, **kwargs):
        super().__init__(scatter_op="indexing", **kwargs)


class ScatterSum(_Scatter):
    def __init__(self, **kwargs):
        super().__init__(scatter_op="add", **kwargs)


class ScatterMax(_Scatter):
    def __init__(self, **kwargs):
        super().__init__(scatter_op="max", **kwargs)
