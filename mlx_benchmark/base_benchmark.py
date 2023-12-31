import multiprocessing as mp
from time import time
from typing import Tuple

import mlx.core as mx
import numpy as np
import torch


class BaseBenchmark:
    """
    Base class for benchmarking different layers or models.
    """

    def __init__(self, **kwargs):
        self.device = None
        self.y = []
        self.args_str = " ".join([f"{k[:3]}={v}" for k, v in kwargs.items()])

    def setup(self, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Abstract method for setting up data before forward pass.
        This is here that data is loaded and optionally preprocessed.

        Returns 2 np.ndarray, the first is the input for the mlx forward
        and the second is for the torch forward.
        """
        raise NotImplementedError

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

    def run(
        self, framework, device=None, iterations=10, **kwargs
    ) -> Tuple[float, float]:
        """
        Runs the benchmark for a specified number of iterations.
        Measures and records the duration of each forward pass.
        """
        data_mlx, data_torch = self.setup(**kwargs)

        if framework == "mlx":
            data = mx.array(data_mlx)
            forward_fn = self.forward_mlx

        elif framework == "torch":
            self.device = device
            data = torch.tensor(data_torch, device=self.device)
            forward_fn = self.forward_torch

        else:
            raise ValueError("Invalid framework.")

        # Measures runtime for n iterations.
        mean_duration = np.mean(
            [
                self._measure_runtime(forward_fn, data, **kwargs)
                for _ in range(iterations + 1)
            ][1:]
        )
        # [1:] is used to remove the first measure, usually slower
        # due to cold start.

        mean_duration_ms = np.round(mean_duration * 1000)
        return mean_duration_ms

    def _measure_runtime(self, fn, data, **kwargs) -> float:
        """
        Simply runs the forward method and measures metrics.
        """
        tic = time()
        fn(data, **kwargs)

        duration = time() - tic

        return duration

    def sync_mps_if_needed(self):
        """
        Call this function after every torch implementation to ensure
        the mps execution has finished.
        """
        if self.device == torch.device("mps"):
            torch.mps.synchronize()
