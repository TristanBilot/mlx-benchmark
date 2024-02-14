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


def calculate_speedup(a, compared_to):
    percentage_difference = -((a - compared_to) / a)
    return percentage_difference * 100


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
        if args.compile:
            headers.append("mlx_gpu_compile")
        if args.include_cpu:
            headers.append("mlx_cpu")
    if args.include_mps:
        headers.append("mps")
    if args.include_cpu:
        headers.append("cpu")
    if args.include_cuda:
        headers.append("cuda")

    if args.compile and args.include_mlx:
        h = "mlx_gpu_compile/mlx_gpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu_compile"], compared_to=v["mlx_gpu"])

    if args.include_mps and args.include_mlx:
        h = "mlx_gpu/mps speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["mps"])

    if args.include_cpu and args.include_mlx:
        h = "mlx_gpu/mlx_cpu speedup"
        headers.append(h)
        for k, v in times.items():
            v[h] = calculate_speedup(v["mlx_gpu"], compared_to=v["mlx_cpu"])

    if args.include_cpu and args.include_cuda:
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
        lambda x, rounding: f"{'+' if x > 0 else ''}{int(x) if rounding == 0 else round(x, rounding)}"
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
