# mlx-benchmark
This repo aims to benchmark Apple's MLX operations and layers, on all Apple Silicon chips, along with some GPUs.

**Contribute:** If you have a device not yet featured in the benchmark, especially the ones listed below, your PR is welcome to broaden the scope and accuracy of this project.

Current devices: `M1`, `M1 Pro`, `M2`, `M2 Pro`, `M2 Max`, `M2 Ultra`, `M3 Pro`.

Missing devices: `M1 Max`, `M1 Ultra`, `M3`, `M3 Ultra`, and `other CUDA GPUs`.

## Benchmarks

Benchmarks are generated by measuring the runtime of every `mlx` operations, along with their equivalent in pytorch with `mps`, `cpu` and `cuda` backends. For each operation, we measure the runtime of multiple experiments. We propose 2 benchmarks based on these experiments:

* [Detailed benchmark](benchmarks/average_benchmark.md): provides the runtime of each experiment. 
* [Average runtime benchmark](benchmarks/detailed_benchmark.md): computes the mean of experiments. Easier to navigate, with fewer details.


## Installation


### Installation on Mac devices

Running the benchmark locally is straightforward. Create a new env with `osx-arm64` architecture and install the dependencies.

```shell
CONDA_SUBDIR=osx-arm64 conda create -n mlx_benchmark python=3.10 numpy pytorch torchvision scipy requests -c conda-forge

pip install -r requirements.txt
```


### Installation on other devices
Other operating systems than macOS can only run the torch experiments, on CPU or with a CUDA device. Install a new env without the `CONDA_SUBDIR=osx-arm64` prefix and install the torch package that matches your CUDA version. Then install all the requirements within `requirements.txt`, except `mlx`.

Finally, open the `config.py` file and set:
```
USE_MLX = False
```
to avoid importing the mlx package, which cannot be installed on non-Mac devices.

## Run the benchmark

### Run on Mac

To run the benchmark on mps, mlx and CPU:

```shell
python run_benchmark.py --include_mps=True --include_mlx=True --include_cpu=True
```


### Run on other devices

To run the torch benchmark on CUDA and CPU:

```shell
python run_benchmark.py --include_mps=False --include_mlx=False --include_cuda=True --include_cpu=True
```

## Contributing

Everyone can contribute to the benchmark! If you have a missing device or if you want to add a missing layer/operation, please read the [contribution guidelines](CONTRIBUTING.md).
