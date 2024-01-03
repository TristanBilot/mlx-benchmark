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
python run_benchmark.py --include_mps=False --include_mlx=False --include_cuda=True --include_cpu=True
```

Once run, 2 tables will be printed. Copy-paste the detailed benchmark into [detailed_benchmark.md](benchmarks/detailed_benchmark.md) and do the same for the average benchmark into [average_benchmark.md](benchmarks/average_benchmark.md). You can then submit a pull request. To ensure consistency in the results, ensure that enough memory is available before running the benchmarks.

## Add a new layer/operation

Many layers and basic operations are still missing in the benchmark. New examples can be easily added 