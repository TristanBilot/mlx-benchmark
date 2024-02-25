from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark

if USE_MLX:
    mlx_simple_operations = {
        "sort": mx.sort,
        "argmax": mx.argmax,
        "softmax": mx.softmax,
        "relu": mx_nn.relu,
        "leaky_relu": mx_nn.leaky_relu,
        "prelu": mx_nn.prelu,
        "softplus": mx_nn.softplus,
        "selu": mx_nn.selu,
        "sigmoid": mx.sigmoid,
        "sum": mx.sum,
    }

torch_simple_operations = {
    "sort": torch.sort,
    "argmax": torch.argmax,
    "softmax": F.softmax,
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "prelu": F.prelu,
    "softplus": F.softplus,
    "selu": F.selu,
    "sigmoid": F.sigmoid,
    "sum": torch.sum,
}

if USE_MLX:
    assert (
        torch_simple_operations.keys() == mlx_simple_operations.keys()
    ), "torch and mlx operations are not the same."
    simple_operations = {
        op: [mlx_simple_operations[op], torch_simple_operations[op]]
        for op in torch_simple_operations.keys()
    }
else:
    simple_operations = {
        op: [None, torch_simple_operations[op]] for op in torch_simple_operations.keys()
    }


class SimpleOperationBenchmark(BaseBenchmark):
    """
    This class can be overridden for simple operations that only necessitate the
    default inputs and an optional `axis` arg.
    """

    def __init__(self, basic_operation, **kwargs):
        super().__init__(**kwargs)

        self.basic_operation = basic_operation

    def forward_mlx(self, compile=False, **kwargs):
        fn = simple_operations[self.basic_operation][0]

        if "axis" in self.kwargs:
            kwargs["axis"] = self.kwargs["axis"]

        fn = self.compile_if_needed(fn, compile=compile)
        y = fn(*self.inputs)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        fn = simple_operations[self.basic_operation][1]

        kwargs = {}
        if "axis" in self.kwargs:
            kwargs["dim"] = self.kwargs["axis"]

        y = fn(*self.inputs, **kwargs)
        self.sync_torch_gpu_if_needed()


class Sort(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("sort", **kwargs)


class Argmax(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("argmax", **kwargs)


class Softmax(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("softmax", **kwargs)


class ReLU(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("relu", **kwargs)


class LeakyReLU(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("leaky_relu", **kwargs)


class PReLU(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("prelu", **kwargs)


class Softplus(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("softplus", **kwargs)


class SeLU(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("selu", **kwargs)


class Sigmoid(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("sigmoid", **kwargs)


class Sum(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("sum", **kwargs)


class SumAll(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("sum", **kwargs)
