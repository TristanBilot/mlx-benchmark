import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

from argparse import ArgumentParser
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch
from tqdm import tqdm

from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

from utils import print_benchmark
from operations import *


def run_processes(operations, backends, iterations=5):
    """
    Runs all operations, on all backends, for a specified number of iterations.
    """
    all_times = defaultdict(dict)

    for i, backend in enumerate(backends):
        print(f"\nRunning benchmarks on {backend} ({i + 1}/{len(backends)})")
        if "mlx" in backend:
            times = run_mlx_backend(operations, backend, iterations)
        else:
            times = run_backend(operations, backend, iterations)
        for op_name, duration in times.items():
            all_times[op_name][backend] = duration

    print("\nDetailed benchmark:")
    print_benchmark(all_times, backends)
    print("\n Average benchmark:")
    print_benchmark(all_times, backends, reduce_mean=True)


def run_mlx_backend(operations, backend, iterations, ops_per_process=10):
    """
    Runs all operations on the given backend in a separate process, to reduce memory requirements.

    ``ops_per_process`` determines the number of ops to run in each process.
    Decreasing this number decreases the max memory usage.
    """
    times = {}

    with tqdm(total=len(operations)) as pbar:
        for i in range(0, len(operations), ops_per_process):
            queue = mp.Queue()
            p = mp.Process(
                target=run_backend,
                args=(
                    operations[i : min(i + 10, len(operations))],
                    backend,
                    iterations,
                    queue,
                ),
            )
            p.start()
            p.join()
            times.update(queue.get())
            queue.close()
            pbar.update(min(10, len(operations) - i))

    return times


def run_backend(operations, backend, iterations, queue=None):
    """
    Runs all operations on the given backend for a specified number of iterations.
    """
    times = {}

    op_iterable = tqdm(operations) if queue is None else operations

    for op in op_iterable:
        op_name = type(op).__name__ + " / " + op.args_str
        duration = run(op, backend, iterations)
        times[op_name] = duration

    if queue is None:
        return times
    queue.put(times)


def run(op, backend, iterations):
    """
    Measures runtime of a single op on the given framework and device specified by backend.
    """

    if backend == "mlx_gpu":
        mx.set_default_device(mx.gpu)
        duration = np.mean([op.run(framework="mlx") for _ in range(iterations)])
    elif backend == "mlx_gpu_compile":
        mx.set_default_device(mx.gpu)
        duration = np.mean(
            [op.run(framework="mlx", compile=True) for _ in range(iterations)]
        )
    elif backend == "mlx_cpu":
        mx.set_default_device(mx.cpu)
        duration = np.mean([op.run(framework="mlx") for _ in range(iterations)])
    elif backend == "cpu":
        duration = np.mean(
            [op.run(framework="torch", device="cpu") for _ in range(iterations)]
        )
    elif backend == "mps":
        duration = np.mean(
            [op.run(framework="torch", device="mps") for _ in range(iterations)]
        )
    elif backend == "cuda":
        duration = np.mean(
            [op.run(framework="torch", device="cuda") for _ in range(iterations)]
        )

    op.clear()

    if backend in ["mps", "cuda"]:
        torch.mps.empty_cache()
        torch.cuda.empty_cache()

    return duration


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--include_mlx_gpu", type=strtobool, default="True")
    parser.add_argument("--include_mlx_gpu_compile", type=strtobool, default="True")
    parser.add_argument("--include_mlx_cpu", type=strtobool, default="True")
    parser.add_argument("--include_mps", type=strtobool, default="True")
    parser.add_argument("--include_cpu", type=strtobool, default="True")
    parser.add_argument("--include_cuda", type=strtobool, default="False")
    args = parser.parse_args()
    print(args)
    print(f"Use MLX: {USE_MLX}")

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."
    if args.include_cuda:
        assert torch.cuda.is_available(), "CUDA device not found."

    backends = [
        arg.replace("include_", "") for arg, value in vars(args).items() if value
    ]

    operations = [
        Argmax(dim1="64x1024x128", axis=0),
        Argmax(dim1="64x1024x128", axis=1),
        Argmax(dim1="64x1024x128", axis=2),
        Argmax(dim1="64x128x1024", axis=2),
        BCE(dim1="1000000", dim2="1000000"),
        BCE(dim1="100000x32", dim2="100000x32"),
        BCE(dim1="100000x64x2", dim2="100000x64x2"),
        BCE(dim1="128x100000", dim2="128x100000"),
        Concat(dim1="1000000x64", dim2="1000000x32", axis=1),
        Concat(dim1="1000000x64", dim2="1000000x128", axis=1),
        Concat(dim1="1000000x64", dim2="1000000x64", axis=0),
        Concat(dim1="64x1000000", dim2="64x1000000", axis=0),
        Conv1d(dim1="100x256x3", dim2="8x3x3"),
        Conv1d(dim1="100x256x256", dim2="8x3x256"),
        Conv1d(dim1="16x1000x80", dim2="128x11x80"),
        Conv1d(dim1="16x1000x3", dim2="128x11x3"),
        Conv2d(dim1="100x256x256x3", dim2="8x3x3x3"),
        Conv2d(dim1="10x256x256x12", dim2="8x3x3x12"),
        Conv2d(dim1="1x256x256x128", dim2="8x3x3x128"),
        Conv2d(dim1="100x28x28x3", dim2="8x3x3x3"),
        Conv2d(dim1="1000x28x28x3", dim2="8x3x3x3"),
        Gather(dim1="64x256", dim2="10"),
        Gather(dim1="64x256", dim2="1000"),
        Gather(dim1="64x256", dim2="1000000"),
        Gather(dim1="1024x32", dim2="10"),
        Gather(dim1="1024x32", dim2="1000"),
        Gather(dim1="1024x32", dim2="1000000"),
        LeakyReLU(dim1="128x16x1024"),
        LeakyReLU(dim1="64x128x1024"),
        Linear(dim1="100x1024x32", dim2="32x1024", dim3="1024"),
        Linear(dim1="100x1024x64", dim2="64x1024", dim3="1024"),
        Linear(dim1="100x1024x256", dim2="256x1024", dim3="1024"),
        Linear(dim1="100x1024x512", dim2="512x1024", dim3="1024"),
        Linear(dim1="100x1x51200", dim2="51200x1", dim3="1"),
        MatMul(dim1="32x1x1000", dim2="32x1000x128"),
        MatMul(dim1="1000x64x256", dim2="256x32"),
        MatMul(dim1="1000x64x1024", dim2="1000x1024x32"),
        MatMul(dim1="1000x1024x64", dim2="1000x64x256"),
        MatMul(dim1="64x1000000", dim2="1000000x32"),
        MatMul(dim1="1000000x64", dim2="64x1024"),
        PReLU(dim1="128x16x1024", dim2="1"),
        PReLU(dim1="64x128x1024", dim2="1"),
        ReLU(dim1="128x16x1024"),
        ReLU(dim1="64x128x1024"),
        Scatter(dim1="64x16", dim2="10"),
        Scatter(dim1="64x16", dim2="1000"),
        Scatter(dim1="64x16", dim2="1000000"),
        Scatter(dim1="1024x32", dim2="10"),
        Scatter(dim1="1024x32", dim2="1000"),
        Scatter(dim1="1024x32", dim2="1000000"),
        ScatterSum(dim1="64x16", dim2="10"),
        ScatterSum(dim1="64x16", dim2="1000"),
        ScatterSum(dim1="64x16", dim2="1000000"),
        ScatterSum(dim1="1024x32", dim2="10"),
        ScatterSum(dim1="1024x32", dim2="1000"),
        ScatterSum(dim1="1024x32", dim2="1000000"),
        ScatterMax(dim1="64x16", dim2="10"),
        ScatterMax(dim1="64x16", dim2="1000"),
        ScatterMax(dim1="64x16", dim2="1000000"),
        ScatterMax(dim1="1024x32", dim2="10"),
        ScatterMax(dim1="1024x32", dim2="1000"),
        ScatterMax(dim1="1024x32", dim2="1000000"),
        SeLU(dim1="128x16x1024"),
        SeLU(dim1="64x128x1024"),
        Sigmoid(dim1="128x16x1024"),
        Sigmoid(dim1="64x128x1024"),
        Softmax(dim1="64x1000000", axis=-1),
        Softmax(dim1="1000000x64", axis=-1),
        Softmax(dim1="64x16x32x1024", axis=-1),
        Softmax(dim1="128x16x32x1024", axis=-1),
        Softmax(dim1="1024x16x32x128", axis=-1),
        Softmax(dim1="1024x64x32x8", axis=-1),
        Softplus(dim1="128x16x1024"),
        Softplus(dim1="64x128x1024"),
        Sort(dim1="64x128x1024", axis=0),
        Sort(dim1="64x128x1024", axis=1),
        Sort(dim1="64x128x1024", axis=2),
        Sum(dim1="64x128x128x128", axis=0),
        Sum(dim1="64x128x128x128", axis=1),
        Sum(dim1="64x128x128x128", axis=2),
        Sum(dim1="64x128x128x128", axis=3),
        SumAll(dim1="64x128x128x128"),
        SumAll(dim1="1000000"),
        SumAll(dim1="1000000x128"),
        SumAll(dim1="128x1000000"),
    ]

    run_processes(operations, backends)
