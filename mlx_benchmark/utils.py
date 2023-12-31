import torchvision
import torchvision.transforms as transforms


def load_mnist(path="data/mnist/"):
    transform = transforms.Compose([transforms.ToTensor()])

    # Only load the train set lcoally.
    data = torchvision.datasets.MNIST(
        root=path, train=True, download=True, transform=transform
    )
    return data


def print_benchmark(times, args):
    times = dict(times)

    # Column headers.
    headers = []
    if args.include_cpu:
        headers.append("cpu")
    if args.include_mps:
        headers.append("mps")
    if args.include_mlx:
        headers.append("mlx")
    if args.include_cuda:
        headers.append("cuda")

    max_name_length = max(len(name) for name in times.keys())

    header_row = "Layer" + " " * (max_name_length - 5) + " | " + " | ".join(headers)
    print(header_row)

    print("-" * len(header_row))

    for layer, times in times.items():
        times_str = " | ".join(f"{times[header]:.2f}" for header in headers)
        print(f"{layer.ljust(max_name_length)} | {times_str}")
