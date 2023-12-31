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

    padded_headers = [f"{h:<4}" for h in headers]
    header_row = (
        "Layer" + " " * (max_name_length - 5) + " | " + " | ".join(padded_headers)
    )
    header_line = "-" * len(header_row)

    print(header_row)

    prev_layer = ""
    for layer, times in times.items():
        times_str = " | ".join(f"{round(times[header],2):>4}" for header in headers)

        if layer[:5] != prev_layer[:5]:
            print(header_line)
        prev_layer = layer

        print(f"{layer.ljust(max_name_length)} | {times_str}")
