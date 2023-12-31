# mlx-benchmark
Updated repo for benchmarking all MLX layers.


## Installation

### Mac devices
```shell
CONDA_SUBDIR=osx-arm64 conda create -n mlx_benchmark python=3.10 numpy pytorch torchvision scipy requests -c conda-forge

pip install -r requirements.txt
```

### Run benchmark on Mac

To run the benchmark on mps, mlx and CPU:

```shell
python run_benchmark.py --include_mps=True --include_mlx=True --include_cpu=True
```

### Other devices (non-mlx & non-mps)
Other operating systems than macOS can only run the torch experiments, on CPU or with a CUDA device. Install a new env without the `CONDA_SUBDIR=osx-arm64` prefix and install the torch package that matches your CUDA version. Install all the requirements within `requirements.txt`, except `mlx`.

Then open the `config.py` file and set:
```
USE_MLX = False
```

#### Run benchmark on other devices

To run the torch benchmark on CUDA and CPU:

```shell
python run_benchmark.py --include_mps=False --include_mlx=False --include_cuda=True --include_cpu=True
```


