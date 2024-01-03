# Detailed MLX benchmark

Detailed runtime benchmark of mlx operations, measured in `milliseconds`.

* `cpu`: cpu backend with torch framework
* `mps`: mps backend with torch framework
* `mlx`: gpu backend with mlx framework
* `mps/mlx speedup`: runtime speedup of mlx compared to mps (%)
* `cpu/mlx speedup`: runtime speedup of mlx compared to cpu (%)

## Apple Silicon

**M1 Pro**

| Layer                                           | cpu | mps | mlx | mps/mlx speedup (%) | cpu/mlx speedup (%) |
|-----------------------------------------------------|------|------|------|-------------------|-------------------|
| MatMul / dim=32x1x1000 dim=32x1000x128          |   0.80 |   0.46 |   0.35 |  31.43 | 128.57 |
| MatMul / dim=1000x64x256 dim=256x32             |  25.91 |   3.02 |   0.68 | 344.12 | 3710.29 |
| MatMul / dim=1000x64x1024 dim=1000x1024x32      | 633.16 |  13.70 |   3.09 | 343.37 | 20390.61 |
| MatMul / dim=1000x1024x64 dim=1000x64x256       | 1324.61 |   9.91 |   9.84 |   0.71 | 13361.48 |
| MatMul / dim=64x1000000 dim=1000000x32          | 658.85 |  20.82 |  45.39 | -54.13 | 1351.53 |
| MatMul / dim=1000000x64 dim=64x1024             | 1855.61 |  97.79 |  38.59 | 153.41 | 4708.53 |
| Softmax / dim=64x1000000                        |  35.01 |   6.10 |   4.57 |  33.48 | 666.08 |
| Softmax / dim=1000000x64                        |  37.24 |   9.59 |   3.66 | 162.02 | 917.49 |
| Softmax / dim=64x16x32x1024                     |  19.09 |   4.38 |   1.77 | 147.46 | 978.53 |
| Softmax / dim=128x16x32x1024                    |  34.98 |   8.46 |   3.15 | 168.57 | 1010.48 |
| Softmax / dim=1024x16x32x128                    |  34.31 |  10.19 |   3.15 | 223.49 | 989.21 |
| Softmax / dim=1024x64x32x8                      |  22.89 |   2.75 |   7.75 | -64.52 | 195.35 |
| Linear / dim=100x1024x32 dim=32x1024 dim=1024   |  48.84 |   7.37 |   8.46 | -12.88 | 477.30 |
| Linear / dim=100x1024x64 dim=64x1024 dim=1024   |  62.82 |  12.49 |  10.42 |  19.87 | 502.88 |
| Linear / dim=100x1024x256 dim=256x1024 dim=1024 | 137.22 |  48.40 |  18.12 | 167.11 | 657.28 |
| Linear / dim=100x1024x512 dim=512x1024 dim=1024 | 247.90 |  95.22 |  30.51 | 212.09 | 712.52 |
| Linear / dim=100x1x51200 dim=51200x1 dim=1      |   0.78 |   0.77 |   0.79 |  -2.53 |  -1.27 |
| Conv2d / dim=100x256x256x3 dim=8x3x3x3          | 147.53 |   7.22 |  38.12 | -81.06 | 287.01 |
| Conv2d / dim=100x256x256x12 dim=8x3x3x12        | 262.34 |  22.22 | 138.33 | -83.94 |  89.65 |
| Conv2d / dim=10x256x256x128 dim=8x3x3x128       | 209.26 |  19.70 | 139.07 | -85.83 |  50.47 |
| Conv2d / dim=100x28x28x3 dim=8x3x3x3            |   1.49 |   0.62 |   1.85 | -66.49 | -19.46 |
| Conv2d / dim=1000x28x28x3 dim=8x3x3x3           |   8.45 |   2.71 |   4.95 | -45.25 |  70.71 |
| BCE / dim=1000000 dim=1000000                   |   1.32 |   1.28 |   0.80 |  60.00 |  65.00 |
| BCE / dim=100000x32 dim=100000x32               |   3.44 |   3.76 |   1.94 |  93.81 |  77.32 |
| BCE / dim=100000x64x2 dim=100000x64x2           |  14.55 |  14.61 |   7.04 | 107.53 | 106.68 |
| BCE / dim=128x100000 dim=128x100000             |  14.62 |  14.82 |   7.05 | 110.21 | 107.38 |
| Concat / dim=1000000x64 dim=1000000x32 axi=1    |  33.01 |   4.50 |   4.42 |   1.81 | 646.83 |
| Concat / dim=1000000x64 dim=1000000x128 axi=1   |  47.96 |   9.29 |   8.63 |   7.65 | 455.74 |
| Concat / dim=1000000x64 dim=1000000x64 axi=0    |  45.09 |   5.82 |   6.12 |  -4.90 | 636.76 |
| Concat / dim=64x1000000 dim=64x1000000 axi=0    |  43.53 |   5.89 |   5.82 |   1.20 | 647.94 |
