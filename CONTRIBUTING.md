# Contributing to mlx-benchmark

This repo could not exist without the contributions from everybody. There are two main ways to contribute to this project:

## Add a new device
If you have an Apple Silicon chip that is missing in the benchmark, or a popular CUDA GPU not yet benchmarked on torch benchmarks, you can add your own experimental results by running the bench locally on your device.

Follow the [installation instructions](README.md#installation) and run the benchmark on `mps`, `mlx` and `cpu` devices if you propose a Mac-based benchmark. This is the default behavior when running the benchmark:

```shell
python run_benchmark.py
```

If you propose a CUDA GPU-based benchmark, running the benchmark on `cpu` and `cuda` devices is enough:

```shell
python run_benchmark.py --include_mps=False --include_mlx_cpu=False --include_mlx_gpu=False --include_mlx_gpu_compile=False --include_cuda=True
```

Once run, 2 tables will be printed. Copy-paste the detailed benchmark into [detailed_benchmark.md](benchmarks/detailed_benchmark.md) and do the same for the average benchmark into [average_benchmark.md](benchmarks/average_benchmark.md). You can then submit a pull request. To ensure consistency in the results, ensure that enough memory is available before running the benchmarks.

Before submitting your PR, ensure to add the config of your M ship along with the version of MLX you're using. This can be easily done by running:

```python
python mlx_benchmark/get_cpu_gpu_config.py

>>> (Apple M1 Pro: 2E+8P+16GPU+16GB) - mlx: 0.2.0
```

## Add a new operation

Many layers and basic operations are still missing in the benchmark. New examples can be easily added to the benchmark, we take here the example of the `concat` operation.

1. Append your benchmarks to the existing ones within `run_benchmark.py`.

```python
operations = [
    ...
    Concat(dim1="1000000x64", dim2="1000000x32", axis=1),
    Concat(dim1="1000000x64", dim2="1000000x128", axis=1),
    Concat(dim1="1000000x64", dim2="1000000x64", axis=0),
    Concat(dim1="64x1000000", dim2="64x1000000", axis=0),
]
```
The arguments starting with `dim*` will create an input tensor of the given shape. If multiple dims are given, like in the previous example, the input tensors for the benchmark can be accessed using `self.inputs[0]` and `self.inputs[1]`. All other arguments such as `axis` can be accessed using `self.kwargs["axis"]`.

2. Create a new file and write the actual implementation of the operation, here in `operations/concat.py`.

```python
from config import USE_MLX

if USE_MLX:
    import mlx.core as mx

import torch

from base_benchmark import BaseBenchmark


class Concat(BaseBenchmark):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_mlx(self, **kwargs):
        a, b = self.inputs

        y = mx.concatenate([a, b], axis=self.kwargs["axis"])
        mx.eval(y)

    @torch.no_grad()
    def forward_torch(self, **kwargs):
        a, b = self.inputs

        y = torch.cat([a, b], dim=self.kwargs["axis"])
        self.sync_mps_if_needed()
```

The structure is almost always the same for all operations. The method `forward_mlx` is the actual implementation of the mlx operation, and the same applies for `forward_torch`. For the mlx implementation, `mx.eval(.)` should be used to compute the operation, whereas `self.sync_mps_if_needed()` should be used after the torch operation.

If the default inputs provided within the args are not enough to implement the new benchmark, you can add your own attributes by overriding this method:

```python
def additional_preprocessing(self, framework):
    if framework == "mlx":
        self.specific_input_for_mlx = ...

def forward_mlx(self, **kwargs):
    a = self.specific_input_for_mlx
    ...
```

3. Lastly, append the new operation in `operations/__init__.py`:

```python
...
from .concat import Concat
```

For simpler operations that only require the default input tensors given by the `dim` provided in args, you can implement the `SimpleOperationBenchmark` class in the same way as done in `simple_operations.py`.

## New features

Enhancements and new features are always welcome! Feel free to submit issues or pull requests.
