import gc
import multiprocessing as mp
from argparse import ArgumentParser
from collections import defaultdict
from distutils.util import strtobool

import torch
from tqdm import tqdm

from utils import print_benchmark
from layers import (
    Linear,
    Conv2d,
    MatMul,
    Softmax,
    BCE,
)


def run_processes(layers, args):
    """
    Runs all layers (i.e. operations) in serial, on separate processes.
    Using processes avoids exploding memory within the main process during the bench.
    """
    all_times = defaultdict(dict)
    queue = mp.Queue()

    for layer in tqdm(layers, total=len(layers)):
        p = mp.Process(target=run, args=(layer, args, queue))
        p.start()

        times = queue.get()
        p.join()

        # NOTE: without this, memory still increases until the end of the bench.
        del layer
        gc.collect()

        all_times.update(times)

    print_benchmark(all_times, args)


def run(layer, args, queue=None):
    """
    Measures runtime of a single layer on all frameworks and devices included in args.
    """
    times = times = defaultdict(dict)
    layer_name = type(layer).__name__ + " / " + layer.args_str

    # MLX benchmark.
    if args.include_mlx:
        mlx_time = layer.run(framework="mlx")
        times[layer_name]["mlx"] = mlx_time

    # CPU PyTorch benchmarks.
    if args.include_cpu:
        cpu_time = layer.run(framework="torch", device=torch.device("cpu"))
        times[layer_name]["cpu"] = cpu_time

    # MPS PyTorch benchmarks.
    if args.include_mps:
        mps_time = layer.run(framework="torch", device=torch.device("mps"))
        times[layer_name]["mps"] = mps_time

    # CUDA PyTorch benchmark.
    if args.include_cuda:
        cuda_time = layer.run(framework="torch", device=torch.device("cuda"))
        times[layer_name]["cuda"] = cuda_time

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

    if args.include_cuda:
        assert torch.cuda.is_available(), "CUDA device not found."

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

    layers = [
        Softmax(dim="64x1000000"),
        Softmax(dim="1000000x64"),
        Softmax(dim="64x16x32x1024"),
        Softmax(dim="128x16x32x1024"),
        Softmax(dim="1024x16x32x128"),
        Softmax(dim="1024x64x32x8"),
        Linear(hid=32),
        Linear(hid=64),
        Linear(hid=128),
        Linear(hid=256),
        Conv2d(hid=32, channels=2),
        Conv2d(hid=32, channels=10),
        Conv2d(hid=32, channels=100),
        Conv2d(hid=64, channels=2),
        Conv2d(hid=64, channels=10),
        Conv2d(hid=64, channels=100),
        MatMul(dim="64x256"),
        MatMul(dim="1000x64x256"),
        MatMul(dim="1000x64x1024"),
        MatMul(dim="1000x1024x64"),
        MatMul(dim="10x100x64x1024"),
        BCE(dim="1000000"),
        BCE(dim="100000x32"),
        BCE(dim="100000x64x2"),
        BCE(dim="128x100000"),
    ]

    run_processes(layers, args)
