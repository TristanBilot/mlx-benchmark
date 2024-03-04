import time
from typing import Callable

from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import numpy as np
import torch


class BaseBenchmark:
    """
    Base class for benchmarking different operations or layers.
    """

    def __init__(self, **kwargs):
        self.device = None
        self.y = []
        self.args_str = " ".join([f"{k[:3]}={v}" for k, v in kwargs.items()])

        self.kwargs = kwargs
        self.compiled_fn: Callable = None

        self.inputs = None

    def compute_inputs(self, framework, device=None):
        """
        Generates the default inputs for all benchmarks.
        """
        dims = []
        for d in ["dim1", "dim2", "dim3"]:
            if d in self.kwargs:
                dim = self.kwargs[d].split("x")
                dims.append([int(i) for i in dim])

        if framework == "mlx":
            self.inputs = [mx.random.normal(dim).astype(mx.float32) for dim in dims]
        else:
            self.inputs = [
                torch.randn(dim, device=device, dtype=torch.float32) for dim in dims
            ]

        if "axis" in self.kwargs:
            self.axis = self.kwargs["axis"]

    def additional_preprocessing(self, framework=None, device=None):
        """
        Can be overridden if custom preprocessing has to be performed on the default
        `self.inputs` given to operations.
        """
        pass

    def forward_mlx(self, **kwargs):
        """
        Abstract method for the forward pass of the benchmark on MLX.
        Should be implemented by subclasses to define the actual computation
        or model inference to be benchmarked.
        """
        raise NotImplementedError

    def forward_torch(self, **kwargs):
        """
        Abstract method for the forward pass of the benchmark on torch.
        Same implementation in torch.
        """
        raise NotImplementedError

    def run(self, framework, compile=False, device=None, **kwargs) -> float:
        """
        Runs the benchmark for a specified number of iterations.
        Measures and records the duration of each forward pass.
        """
        if self.inputs is None:
            self.compute_inputs(framework, device)
            self.additional_preprocessing(framework, device)

        if framework == "mlx":
            forward_fn = self.forward_mlx
            kwargs = {
                **kwargs,
                "compile": compile,
            }

        elif framework == "torch":
            self.device = device
            forward_fn = self.forward_torch

        else:
            raise ValueError("Invalid framework.")

        # Measures runtime for n iterations.
        duration = np.mean(
            [self._measure_runtime(forward_fn, **kwargs) for _ in range(10)][1:]
        )
        # [1:] is used to remove the first measure, usually slower
        # due to cold start.

        duration_ms = duration * 1000
        return duration_ms

    def _measure_runtime(self, fn, **kwargs) -> float:
        """
        Simply runs the forward method and measures metrics.
        """
        tic = time.perf_counter()
        try:
            fn(**kwargs)
        except (NotImplementedError, RuntimeError):
            return float("nan")

        duration = time.perf_counter() - tic

        return duration

    def sync_torch_gpu_if_needed(self):
        """
        Call this function after every torch implementation to ensure
        the mps or cuda execution has finished.
        """
        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def compile_if_needed(self, fn, **kwargs):
        """
        Caches the compiled function `fun` when called for the first time.
        """
        if "compile" in kwargs and kwargs["compile"]:
            if self.compiled_fn is not None:
                return self.compiled_fn
            self.compiled_fn = mx.compile(fn)
            return self.compiled_fn
        return fn

    def clear(self):
        self.inputs = None
        self.y = []
        for attribute in [
            "a_torch",
            "b_torch",
            "b_mlx",
            "src",
            "index",
            "node_features",
        ]:
            if hasattr(self, attribute):
                delattr(self, attribute)
