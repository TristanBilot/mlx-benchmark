# Detailed MLX benchmark

Detailed runtime benchmark of mlx operations, measured in `milliseconds`.

* `mlx_gpu`: mlx framework with gpu backend
* `mlx_cpu`: mlx framework with cpu backend
* `cpu`: torch framework with cpu backend
* `mps`: torch framework with mps (gpu) backend
* `mps/mlx_gpu speedup`: runtime speedup of mlx_gpu compared to mps
* `mlx_cpu/mlx_gpu speedup`: runtime speedup of mlx_gpu compared to mlx_cpu

## Apple Silicon

**M1 Pro**

| Operation                                           | mlx_gpu | mlx_cpu | mps | cpu | mps/mlx_gpu speedup | mlx_cpu/mlx_gpu speedup |
|-----------------------------------------------------|-------|-------|------|------|-------------------|-----------------------|
| MatMul / dim=32x1x1000 dim=32x1000x128          |   0.27 |   0.85 |   0.44 |   0.79 |   0.63 |   2.15 |
| MatMul / dim=1000x64x256 dim=256x32             |   0.74 |   1.94 |   1.87 |  29.62 |   1.53 |   1.62 |
| MatMul / dim=1000x64x1024 dim=1000x1024x32      |   2.61 |  18.54 |  13.74 | 418.90 |   4.26 |   6.10 |
| MatMul / dim=1000x1024x64 dim=1000x64x256       |   9.93 |  49.91 |   9.27 | 1125.18 |  -0.07 |   4.03 |
| MatMul / dim=64x1000000 dim=1000000x32          |  42.01 |  10.76 |  19.95 | 228.32 |  -0.53 |  -0.74 |
| MatMul / dim=1000000x64 dim=64x1024             |  37.21 | 138.14 |  93.40 | 1757.58 |   1.51 |   2.71 |
| Softmax / dim=64x1000000                        |   4.52 |  37.53 |   6.36 |  35.28 |   0.41 |   7.30 |
| Softmax / dim=1000000x64                        |   3.66 |  35.76 |   9.04 |  37.04 |   1.47 |   8.77 |
| Softmax / dim=64x16x32x1024                     |   1.70 |  18.67 |   4.44 |  28.46 |   1.61 |   9.98 |
| Softmax / dim=128x16x32x1024                    |   3.15 |  38.04 |   7.81 |  34.17 |   1.48 |  11.08 |
| Softmax / dim=1024x16x32x128                    |   3.53 |  36.85 |   9.51 |  36.34 |   1.69 |   9.44 |
| Softmax / dim=1024x64x32x8                      |   7.37 |  85.73 |   3.39 |  22.47 |  -0.54 |  10.63 |
| Linear / dim=100x1024x32 dim=32x1024 dim=1024   |   8.08 |  26.31 |   7.31 |  46.84 |  -0.10 |   2.26 |
| Linear / dim=100x1024x64 dim=64x1024 dim=1024   |   9.56 |  31.46 |  12.51 |  66.79 |   0.31 |   2.29 |
| Linear / dim=100x1024x256 dim=256x1024 dim=1024 |  17.90 |  47.38 |  48.18 | 131.62 |   1.69 |   1.65 |
| Linear / dim=100x1024x512 dim=512x1024 dim=1024 |  30.39 |  71.05 |  94.78 | 244.28 |   2.12 |   1.34 |
| Linear / dim=100x1x51200 dim=51200x1 dim=1      |   0.63 |   0.37 |   1.02 |   0.73 |   0.62 |  -0.41 |
| Conv2d / dim=100x256x256x3 dim=8x3x3x3          |  34.72 | 965.27 |   6.94 | 138.86 |  -0.80 |  26.80 |
| Conv2d / dim=100x256x256x12 dim=8x3x3x12        | 136.82 | 4198.33 |  21.44 | 264.79 |  -0.84 |  29.69 |
| Conv2d / dim=10x256x256x128 dim=8x3x3x128       | 141.41 | 7242.40 |  19.95 | 213.08 |  -0.86 |  50.22 |
| Conv2d / dim=100x28x28x3 dim=8x3x3x3            |   1.36 |  11.02 |   0.60 |   1.47 |  -0.56 |   7.10 |
| Conv2d / dim=1000x28x28x3 dim=8x3x3x3           |   4.28 | 109.92 |   2.13 |   8.73 |  -0.50 |  24.68 |
| BCE / dim=1000000 dim=1000000                   |   1.00 |   4.28 |   1.26 |   1.31 |   0.26 |   3.28 |
| BCE / dim=100000x32 dim=100000x32               |   1.94 |  12.76 |   3.78 |   3.50 |   0.95 |   5.58 |
| BCE / dim=100000x64x2 dim=100000x64x2           |   7.49 |  49.26 |  14.44 |  14.45 |   0.93 |   5.58 |
| BCE / dim=128x100000 dim=128x100000             |   6.72 |  48.86 |  14.54 |  14.71 |   1.16 |   6.27 |
| Concat / dim=1000000x64 dim=1000000x32 axi=1    |   4.59 |  65.54 |   4.45 |  28.53 |  -0.03 |  13.28 |
| Concat / dim=1000000x64 dim=1000000x128 axi=1   |   8.96 | 147.09 |   9.27 |  62.94 |   0.03 |  15.42 |
| Concat / dim=1000000x64 dim=1000000x64 axi=0    |   5.86 |  64.54 |   5.94 |  41.62 |   0.01 |  10.01 |
| Concat / dim=64x1000000 dim=64x1000000 axi=0    |   5.82 |  84.83 |   6.05 |  42.08 |   0.04 |  13.58 |
