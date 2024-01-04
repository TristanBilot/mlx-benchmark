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
| Argmax     |   1.32 |  13.97 |   1.92 |   8.30 |    +45% |   +957% |
| BCE        |   4.18 |  28.57 |   8.98 |   8.50 |   +114% |   +583% |
| Concat     |   6.51 | 103.22 |   7.14 |  42.93 |     +9% |  +1485% |
| Conv1d     |   3.74 |   3.58 |   2.53 | 152.46 |    -32% |     -4% |
| Conv2d     |  14.79 | 444.64 |   5.05 |  44.68 |    -65% |  +2905% |
| LeakyReLU  |   0.88 |   2.07 |   0.79 |   1.09 |    -10% |   +134% |
| Linear     |  14.77 |  50.73 |  33.53 | 102.24 |   +127% |   +243% |
| MatMul     |  15.59 |  52.57 |  27.29 | 560.23 |    +75% |   +237% |
| PReLU      |   1.42 |   3.46 |   0.79 |   1.46 |    -44% |   +143% |
| ReLU       |   0.54 |   0.72 |   0.73 |   1.29 |    +34% |    +32% |
| SeLU       |   3.11 |  12.86 |   0.73 |   6.67 |    -76% |   +313% |
| Sigmoid    |   0.60 |  32.43 |   0.82 |   6.17 |    +36% |  +5309% |
| Softmax    |   4.00 |  40.28 |  10.90 |  36.93 |   +172% |   +906% |
| Softplus   |   0.77 |  32.68 |   1.16 |   8.79 |    +51% |  +4159% |
| Sort       |   9.53 | 838.22 |  37.53 |  54.08 |   +293% |  +8694% |
| Sum        |   5.76 |  12.76 |   6.38 |  10.80 |    +10% |   +121% |
| SumAll     |   2.62 |   6.75 |   2.87 |   3.43 |     +9% |   +157% |