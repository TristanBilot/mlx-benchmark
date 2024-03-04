import math
from collections import defaultdict

import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


def load_mnist(path="data/mnist/"):
    transform = transforms.Compose([transforms.ToTensor()])

    # Only load the train set lcoally.
    data = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=transform
    )
    return data


def calculate_speedup(a, compared_to):
    percentage_difference = -((a - compared_to) / a)
    return percentage_difference * 100


def print_benchmark(times, backends, reduce_mean=False):
    times = dict(times)

    if reduce_mean:
        new_times = defaultdict(lambda: defaultdict(list))
        for k, v in times.items():
            op = k.split("/")[0]
            for backend, runtime in v.items():
                new_times[op][backend].append(runtime)

        for k, v in new_times.items():
            for backend, runtimes in v.items():
                new_times[k][backend] = np.mean(new_times[k][backend])
        times = new_times

    # Column headers
    header_order = ["mlx_gpu", "mlx_gpu_compile", "mlx_cpu", "mps", "cpu", "cuda"]
    headers = sorted(backends, key=lambda x: header_order.index(x))

    if "mlx_gpu_compile" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu_compile/mlx_gpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu_compile"], compared_to=v["mlx_gpu"])

    if "mps" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu/mps speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["mps"])

    if "mlx_cpu" in backends and "mlx_gpu" in backends:
        h = "mlx_gpu/mlx_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["mlx_cpu"])

    if "cpu" in backends and "cuda" in backends:
        h = "cuda/cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["cuda"], compared_to=v["cpu"])

    max_name_length = max(len(name) for name in times.keys())

    # Formatting the header row
    header_row = (
        "| Operation" + " " * (max_name_length - 5) + " | " + " | ".join(headers) + " |"
    )
    header_line_parts = ["-" * (max_name_length + 6)] + [
        "-" * max(6, len(header)) for header in headers
    ]
    header_line = "|" + "|".join(header_line_parts) + "|"

    print(header_row)
    print(header_line)

    add_plus_symbol = (
        lambda x, rounding: f"{'+' if x > 0 else ''}{(int(x) if not math.isnan(x) else x) if rounding == 0 else round(x, rounding)}"
    )
    format_value = (
        lambda header: f"{add_plus_symbol(times[header], 0):>6}%"
        if "speedup" in header
        else f"{times[header]:>6.2f}"
    )

    for op, times in times.items():
        times_str = " | ".join(format_value(header) for header in headers)

        # Formatting each row
        print(f"| {op.ljust(max_name_length)} | {times_str} |")


def get_dummy_edge_index(shape, num_nodes, device, framework):
    if framework == "mlx":
        import mlx.core as mx
        
        return mx.random.randint(0, num_nodes - 1, shape)
    elif framework == "torch":
        return torch.randint(0, num_nodes - 1, shape).to(device)
    raise ValueError("Framework should be either mlx or torch.")
