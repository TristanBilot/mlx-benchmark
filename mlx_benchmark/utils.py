from collections import defaultdict

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


def print_benchmark(times, args, reduce_mean=False):
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
    headers = []
    if args.include_mlx:
        headers.append("mlx_gpu")
        headers.append("mlx_cpu")
    if args.include_mps:
        headers.append("mps")
    if args.include_cpu:
        headers.append("cpu")
    if args.include_cuda:
        headers.append("cuda")

    if args.include_mps and args.include_mlx:
        h = "mps/mlx_gpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = v["mps"] / v["mlx_gpu"] - 1

    if args.include_cpu and args.include_mlx:
        h = "mlx_cpu/mlx_gpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = v["mlx_cpu"] / v["mlx_gpu"] - 1

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

    for op, times in times.items():
        times_str = " | ".join(f"{times[header]:>6.2f}" for header in headers)

        # Formatting each row
        print(f"| {op.ljust(max_name_length)} | {times_str} |")
