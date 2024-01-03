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

| Operation    | mlx_gpu | mlx_cpu | mps | cpu | mps/mlx_gpu speedup | mlx_cpu/mlx_gpu speedup |
|--------------|-------|-------|------|------|-------------------|-----------------------|
| MatMul   |  15.46 |  36.69 |  23.11 | 593.40 |   0.49 |   1.37 |
| Softmax  |   3.99 |  42.10 |   6.76 |  32.29 |   0.69 |   9.55 |
| Linear   |  13.31 |  35.31 |  32.76 |  98.05 |   1.46 |   1.65 |
| Conv2d   |  63.72 | 2505.39 |  10.21 | 125.39 |  -0.84 |  38.32 |
| BCE      |   4.29 |  28.79 |   8.50 |   8.49 |   0.98 |   5.71 |
| Concat   |   6.31 |  90.50 |   6.43 |  43.79 |   0.02 |  13.35 |
