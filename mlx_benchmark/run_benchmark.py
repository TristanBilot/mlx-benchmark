import multiprocessing as mp

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import gc
from argparse import ArgumentParser
from collections import defaultdict
from distutils.util import strtobool

import torch
from tqdm import tqdm

from config import USE_MLX
from utils import print_benchmark
from operations import *


def run_processes(operations, args):
    """
    Runs all operations (i.e. operations) in serial, on separate processes.
    Using processes avoids exploding memory within the main process during the bench.
    """
    all_times = defaultdict(dict)
    queue = mp.Queue()

    for op in tqdm(operations, total=len(operations)):
        p = mp.Process(target=run, args=(op, args, queue))
        p.start()

        times = queue.get()
        p.join()

        # NOTE: without this, memory still increases until the end of the bench.
        del op
        gc.collect()

        all_times.update(times)

    print("\nDetailed benchmark:")
    print_benchmark(all_times, args)
    print("\n Average benchmark:")
    print_benchmark(all_times, args, reduce_mean=True)


def run(op, args, queue=None):
    """
    Measures runtime of a single op on all frameworks and devices included in args.
    """
    times = times = defaultdict(dict)
    op_name = type(op).__name__ + " / " + op.args_str

    # MLX benchmark.
    if args.include_mlx:
        mlx_time = op.run(framework="mlx")
        times[op_name]["mlx"] = mlx_time

    # CPU PyTorch benchmarks.
    if args.include_cpu:
        cpu_time = op.run(framework="torch", device=torch.device("cpu"))
        times[op_name]["cpu"] = cpu_time

    # MPS PyTorch benchmarks.
    if args.include_mps:
        mps_time = op.run(framework="torch", device=torch.device("mps"))
        times[op_name]["mps"] = mps_time

    # CUDA PyTorch benchmark.
    if args.include_cuda:
        cuda_time = op.run(framework="torch", device=torch.device("cuda"))
        times[op_name]["cuda"] = cuda_time

    if queue is None:
        return times
    queue.put(times)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--include_cpu", type=strtobool, default="True")
    parser.add_argument("--include_mps", type=strtobool, default="True")
    parser.add_argument("--include_mlx", type=strtobool, default="True")
    parser.add_argument("--include_cuda", type=strtobool, default="False")
    args = parser.parse_args()
    print(args)
    print(f"Use MLX: {USE_MLX}")

    if args.include_cuda:
        assert torch.cuda.is_available(), "CUDA device not found."

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

    operations = [
        MatMul(dim1="32x1x1000", dim2="32x1000x128"),
        MatMul(dim1="1000x64x256", dim2="256x32"),
        MatMul(dim1="1000x64x1024", dim2="1000x1024x32"),
        MatMul(dim1="1000x1024x64", dim2="1000x64x256"),
        MatMul(dim1="64x1000000", dim2="1000000x32"),
        MatMul(dim1="1000000x64", dim2="64x1024"),
        Softmax(dim1="64x1000000"),
        Softmax(dim1="1000000x64"),
        Softmax(dim1="64x16x32x1024"),
        Softmax(dim1="128x16x32x1024"),
        Softmax(dim1="1024x16x32x128"),
        Softmax(dim1="1024x64x32x8"),
        Linear(dim1="100x1024x32", dim2="32x1024", dim3="1024"),
        Linear(dim1="100x1024x64", dim2="64x1024", dim3="1024"),
        Linear(dim1="100x1024x256", dim2="256x1024", dim3="1024"),
        Linear(dim1="100x1024x512", dim2="512x1024", dim3="1024"),
        Linear(dim1="100x1x51200", dim2="51200x1", dim3="1"),
        Conv2d(dim1="100x256x256x3", dim2="8x3x3x3"),
        Conv2d(dim1="100x256x256x12", dim2="8x3x3x12"),
        Conv2d(dim1="10x256x256x128", dim2="8x3x3x128"),
        Conv2d(dim1="100x28x28x3", dim2="8x3x3x3"),
        Conv2d(dim1="1000x28x28x3", dim2="8x3x3x3"),
        BCE(dim1="1000000", dim2="1000000"),
        BCE(dim1="100000x32", dim2="100000x32"),
        BCE(dim1="100000x64x2", dim2="100000x64x2"),
        BCE(dim1="128x100000", dim2="128x100000"),
        Concat(dim1="1000000x64", dim2="1000000x32", axis=1),
        Concat(dim1="1000000x64", dim2="1000000x128", axis=1),
        Concat(dim1="1000000x64", dim2="1000000x64", axis=0),
        Concat(dim1="64x1000000", dim2="64x1000000", axis=0),
    ]

    run_processes(operations, args)
