from argparse import ArgumentParser
from collections import defaultdict

import torch
from tqdm import tqdm

from utils import print_benchmark
from layers import (
    Linear,
    Conv2d,
    MatMul,
)


def run(layers, args):
    times = defaultdict(dict)
    for layer in tqdm(layers, total=len(layers)):
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

    print_benchmark(times, args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--include_cpu", type=bool, default=True)
    parser.add_argument("--include_mps", type=bool, default=True)
    parser.add_argument("--include_mlx", type=bool, default=True)
    parser.add_argument("--include_cuda", type=bool, default=False)
    args = parser.parse_args()

    if args.include_cuda:
        assert torch.cuda.is_available(), "CUDA device not found."

    if args.include_mps:
        assert torch.backends.mps.is_available(), "MPS backend not available."

    layers = [
        MatMul(dim="64x256"),
        MatMul(dim="1000x64x256"),
        MatMul(dim="1000x64x1024"),
        MatMul(dim="1000x1024x64"),
        MatMul(dim="10x100x1024x64"),
        MatMul(dim="10x100x64x1024"),
        Linear(hid=32),
        Linear(hid=64),
        Linear(hid=128),
        Conv2d(hid=32, channels=2),
        Conv2d(hid=32, channels=10),
        Conv2d(hid=32, channels=100),
    ]

    run(layers, args)
