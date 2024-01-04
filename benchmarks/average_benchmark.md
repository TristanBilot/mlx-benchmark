# Average MLX benchmark

Averaged runtime benchmark of mlx operations, measured in `milliseconds`.

* `mlx_gpu`: mlx framework with gpu backend
* `mlx_cpu`: mlx framework with cpu backend
* `cpu`: torch framework with cpu backend
* `mps`: torch framework with mps (gpu) backend
* `mps/mlx_gpu speedup`: runtime speedup of mlx_gpu compared to mps
* `mlx_cpu/mlx_gpu speedup`: runtime speedup of mlx_gpu compared to mlx_cpu

## Apple Silicon

**M1 Pro**

| Operation      | mlx_gpu | mlx_cpu | mps | cpu | mps/mlx_gpu speedup | mlx_cpu/mlx_gpu speedup |
|----------------|-------|-------|------|------|-------------------|-----------------------|
| Argmax     |   1.37 |  14.12 |   2.03 |   8.29 |   0.33 |   0.90 |
| BCE        |   4.25 |  28.48 |   9.11 |   8.45 |   0.53 |   0.85 |
| Concat     |   6.20 | 102.03 |   6.83 |  41.29 |   0.09 |   0.94 |
| Conv1d     |   4.18 |   3.59 |   2.50 | 153.66 |  -0.67 |  -0.17 |
| Conv2d     |  64.18 | 2465.32 |  11.20 | 126.76 |  -4.73 |   0.97 |
| LeakyReLU  |   0.98 |   2.04 |   0.74 |   1.27 |  -0.33 |   0.52 |
| Linear     |  14.44 |  51.21 |  33.54 | 100.08 |   0.57 |   0.72 |
| MatMul     |  15.56 |  51.84 |  26.17 | 617.28 |   0.41 |   0.70 |
| PReLU      |   1.38 |   3.36 |   0.73 |   1.33 |  -0.89 |   0.59 |
| ReLU       |   0.55 |   0.72 |   0.71 |   1.14 |   0.23 |   0.24 |
| SeLU       |   2.90 |  12.82 |   0.76 |   6.62 |  -2.84 |   0.77 |
| Sigmoid    |   0.53 |  32.04 |   0.77 |   6.11 |   0.31 |   0.98 |
| Softmax    |   4.00 |  40.18 |  10.56 |  36.13 |   0.62 |   0.90 |
| Softplus   |   0.90 |  32.19 |   1.15 |   8.87 |   0.22 |   0.97 |
| Sort       |   9.47 | 839.95 |  37.56 |  54.57 |   0.75 |   0.99 |
| Sum        |   5.74 |  12.55 |   6.40 |  10.68 |   0.10 |   0.54 |
| SumAll     |   2.66 |   6.79 |   2.78 |   3.49 |   0.04 |   0.61 |