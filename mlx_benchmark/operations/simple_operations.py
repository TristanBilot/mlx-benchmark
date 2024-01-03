from config import USE_MLX

if USE_MLX:
    import mlx.core as mx
    import mlx.nn as mx_nn

import torch
import torch.nn.functional as F

from base_benchmark import BaseBenchmark


simple_operations = {
    "sort": [mx.sort, torch.sort],
    "argmax": [mx.argmax, torch.argmax],
    "softmax": [mx.softmax, F.softmax],
}


class SimpleOperationBenchmark(BaseBenchmark):
    """
    This class can be overridden for simple operations that only necessitate the
    default inputs and an optional `axis` arg.
    """

    def __init__(self, basic_operation, **kwargs):
        super().__init__(**kwargs)

        self.basic_operation = basic_operation

    def forward_mlx(self, **kwargs):
        fn = simple_operations[self.basic_operation][0]

        kwargs = {}
        if "axis" in self.kwargs:
            kwargs["axis"] = self.kwargs["axis"]

        y = fn(*self.inputs, **kwargs)
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        fn = simple_operations[self.basic_operation][1]

        kwargs = {}
        if "axis" in self.kwargs:
            kwargs["dim"] = self.kwargs["axis"]

        y = fn(*self.inputs, **kwargs)
        self.sync_mps_if_needed()


class Sort(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("sort", **kwargs)


class Argmax(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("argmax", **kwargs)


class Softmax(SimpleOperationBenchmark):
    def __init__(self, **kwargs):
        super().__init__("softmax", **kwargs)
