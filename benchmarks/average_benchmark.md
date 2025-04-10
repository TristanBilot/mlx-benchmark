# Average MLX benchmark

Averaged runtime benchmark of mlx operations, measured in `milliseconds`.

* `mlx_gpu`: mlx framework with gpu backend
* `mlx_cpu`: mlx framework with cpu backend
* `cpu`: torch framework with cpu backend
* `mps`: torch framework with mps (gpu) backend
* `mlx_gpu/mps speedup`: runtime speedup of mlx_gpu compared to mps
* `mlx_gpu/mlx_cpu speedup`: runtime speedup of mlx_gpu compared to mlx_cpu
* `cuda/cpu speedup`: runtime speedup of cuda compared to cpu

## Apple Silicon

**M1 (cores: 4E+4P+8GPU)**

| Operation      | mlx_gpu | mlx_cpu | mps | cpu | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|----------------|-------|-------|------|------|-------------------|-----------------------|
| Argmax     |   1.81 |  10.63 |   2.87 |   8.05 |    +58% |   +486% |
| BCE        |   5.51 |  51.81 |  12.19 |  10.87 |   +121% |   +840% |
| Concat     |  19.13 | 100.88 |  19.28 |  49.63 |     +0% |   +427% |
| Conv1d     |   3.83 |   4.53 |   3.73 | 116.13 |     -2% |    +18% |
| Conv2d     |  30.12 | 436.68 |   7.06 |  45.54 |    -76% |  +1349% |
| LeakyReLU  |   2.06 |   2.90 |   1.16 |   1.37 |    -43% |    +41% |
| Linear     |  30.41 |  73.32 |  53.70 | 117.68 |    +76% |   +141% |
| MatMul     |  26.38 |  93.82 |  47.87 | 504.47 |    +81% |   +255% |
| PReLU      |   3.50 |   4.54 |   1.15 |   1.32 |    -67% |    +29% |
| ReLU       |   0.98 |   0.90 |   1.13 |   1.35 |    +14% |     -8% |
| SeLU       |   7.81 |  14.73 |   1.14 |   7.72 |    -85% |    +88% |
| Sigmoid    |   0.96 |  32.66 |   1.16 |   7.23 |    +19% |  +3287% |
| Softmax    |  10.15 |  40.98 |  19.27 |  46.69 |    +89% |   +303% |
| Softplus   |   1.07 |  33.08 |   1.73 |  10.99 |    +60% |  +2977% |
| Sort       |  18.49 | 713.23 |  73.24 |  70.11 |   +296% |  +3756% |
| Sum        |  11.33 |  12.70 |  16.35 |  13.43 |    +44% |    +12% |
| SumAll     |   6.91 |   6.85 |   7.40 |   7.00 |     +7% |      0% |

**M1 Pro (2E+8P+16GPU+16GB) - mlx: 0.5.0**

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.75 |   1.74 |  10.55 |   1.02 |   8.19 |     +0% |    -41% |   +503% |
| BCE         |   2.18 |   0.97 |  59.50 |   0.84 |   8.48 |   +125% |    -61% |  +2629% |
| Concat      |   6.14 |   6.13 |  87.88 |   6.21 |  36.74 |     +0% |     +1% |  +1332% |
| Conv1d      |   1.76 |   1.64 |   3.42 |   1.01 | 154.38 |     +7% |    -42% |    +94% |
| Conv2d      |   5.71 |   5.67 | 443.83 |   2.52 |  42.12 |     +0% |    -55% |  +7669% |
| Gather      |   3.15 |   3.17 |   4.95 |  18.87 |   9.03 |      0% |   +498% |    +57% |
| LeakyReLU   |   0.46 |   0.44 |   0.80 |   0.47 |   1.21 |     +4% |     +2% |    +74% |
| Linear      |   9.57 |   9.76 |  34.65 |  33.21 | 127.82 |     -1% |   +246% |   +261% |
| MatMul      |  10.52 |  10.65 |  38.29 |  22.76 | 498.70 |     -1% |   +116% |   +263% |
| PReLU       |   0.48 |   0.46 |   3.37 |   0.55 |   1.07 |     +3% |    +15% |   +607% |
| ReLU        |   0.47 |   0.43 |   0.63 |   0.55 |   1.08 |     +9% |    +18% |    +34% |
| Scatter     |   0.59 |   0.57 |  30.02 |   3.38 |   1.94 |     +2% |   +473% |  +5002% |
| ScatterSum  |   0.03 |   0.04 |   0.01 |    nan |   1.47 |    -14% |    nan% |    -71% |
| ScatterMax  |   0.03 |   0.04 |   0.01 |    nan |   1.44 |    -10% |    nan% |    -69% |
| SeLU        |   0.51 |   0.46 |   4.86 |   0.47 |   6.72 |    +12% |     -8% |   +849% |
| Sigmoid     |   0.44 |   0.44 |   4.58 |   0.55 |   6.39 |     +0% |    +23% |   +931% |
| Softmax     |   9.44 |   7.32 |  41.66 |   5.96 |  30.23 |    +28% |    -36% |   +341% |
| Softplus    |   0.46 |   0.49 |  35.26 |   0.49 |   8.97 |     -7% |     +6% |  +7646% |
| Sort        |   1.69 |   1.72 | 258.35 |  37.76 |  58.56 |     -1% |  +2129% | +15156% |
| Sum         |   3.38 |   3.46 |   9.25 |   6.06 |  10.02 |     -2% |    +79% |   +173% |
| SumAll      |   2.52 |   2.63 |   6.83 |   2.48 |   3.46 |     -4% |     -1% |   +171% |

**M1 Max (64GB)** mlx 0.2.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   2.14 |   1.69 |  10.80 |   1.93 |   9.17 |    +27% |    -10% |   +403% |
| BCE         |   1.30 |   0.65 |  50.27 |   1.01 |   8.09 |    +98% |    -22% |  +3777% |
| Concat      |   3.20 |   3.20 |  92.35 |   3.27 |  24.79 |     +0% |     +2% |  +2782% |
| Conv1d      |   2.20 |   0.98 |   3.34 |   1.18 | 157.26 |   +124% |    -46% |    +51% |
| Conv2d      |   8.18 |   7.24 | 455.47 |   1.98 |  35.56 |    +13% |    -75% |  +5468% |
| Gather      |   2.51 |   2.37 |   5.94 |   9.78 |   8.92 |     +5% |   +289% |   +136% |
| LeakyReLU   |   0.54 |   0.34 |   4.40 |   0.45 |   0.63 |    +59% |    -15% |   +719% |
| Linear      |   6.73 |   6.49 |  32.46 |  16.44 |  39.44 |     +3% |   +144% |   +382% |
| MatMul      |   4.66 |   4.64 |  47.17 |  11.16 |  88.32 |     +0% |   +139% |   +913% |
| PReLU       |   0.82 |   0.36 |   2.64 |   0.44 |   0.57 |   +127% |    -46% |   +222% |
| ReLU        |   0.36 |   0.33 |   0.82 |   0.44 |   0.60 |     +9% |    +21% |   +125% |
| Scatter     |   4.11 |   4.09 |  30.31 |   1.85 |   1.78 |     +0% |    -55% |   +637% |
| ScatterSum  |   0.05 |   0.03 |   0.01 |    nan |   1.35 |    +42% |    nan% |    -81% |
| ScatterMax  |   0.05 |   0.03 |   0.01 |    nan |   1.35 |    +34% |    nan% |    -81% |
| SeLU        |   1.53 |   0.36 |   7.05 |   0.46 |   5.97 |   +323% |    -69% |   +362% |
| Sigmoid     |   0.38 |   0.36 |  32.57 |   0.50 |   5.43 |     +7% |    +30% |  +8409% |
| Softmax     |   4.84 |   3.71 |  43.48 |   3.88 |  28.93 |    +30% |    -19% |   +798% |
| Softplus    |   0.57 |   0.34 |  32.79 |   0.67 |   8.25 |    +65% |    +17% |  +5642% |
| Sort        |   1.08 |   0.97 | 257.18 |  20.18 |  49.30 |    +10% |  +1773% | +23780% |
| Sum         |   1.75 |   1.74 |   8.87 |   3.10 |  10.35 |     +0% |    +77% |   +406% |
| SumAll      |   1.36 |   1.34 |   6.63 |   1.50 |   3.36 |     +1% |    +10% |   +389% |

**M2 () - mlx 0.2.0**

| Operation      | mlx_gpu | mlx_cpu | mps | cpu | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|----------------|-------|-------|------|------|-------------------|-----------------------|
| Argmax     |   1.71 |  16.21 |   2.71 |   8.51 |    +58% |   +849% |
| BCE        |   3.71 |  82.34 |  13.06 |  13.34 |   +251% |  +2118% |
| Concat     |  12.14 | 161.07 |  12.51 |  46.33 |     +3% |  +1226% |
| Conv1d     |   3.66 |   6.01 |   3.29 | 132.69 |    -10% |    +64% |
| Conv2d     |  27.74 | 705.78 |   5.94 |  56.88 |    -78% |  +2444% |
| LeakyReLU  |   1.50 |   2.44 |   1.10 |   1.33 |    -26% |    +62% |
| Linear     |  25.01 |  99.71 |  57.29 | 183.02 |   +129% |   +298% |
| MatMul     |  22.04 | 120.61 |  78.10 | 629.63 |   +254% |   +447% |
| PReLU      |   2.43 |   4.58 |   1.04 |   1.35 |    -57% |    +88% |
| ReLU       |   0.77 |   1.00 |   1.00 |   1.34 |    +30% |    +29% |
| SeLU       |   5.31 |  17.25 |   1.11 |   8.24 |    -79% |   +224% |
| Sigmoid    |   0.77 |  52.85 |   1.13 |   7.47 |    +47% |  +6797% |
| Softmax    |   7.07 |  65.62 |  14.54 |  60.92 |   +105% |   +828% |
| Softplus   |   0.91 |  53.94 |   1.73 |  12.12 |    +90% |  +5846% |
| Sort       |  16.87 | 1243.25 |  46.69 |  79.31 |   +176% |  +7269% |
| Sum        |   9.15 |  18.38 |  10.47 |  14.19 |    +14% |   +100% |
| SumAll     |   4.31 |   7.79 |   4.96 |   6.11 |    +14% |    +80% |

**M2 Pro (cores: 4E+6P+16GPU)** mlx 0.12.2 torch 2.1.2

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.55 |   1.52 |   9.98 |   1.28 |   7.88 |     +2% |    -17% |   +542% |
| BCE         |   2.03 |   0.82 |  59.53 |   0.74 |   8.01 |   +146% |    -63% |  +2835% |
| Concat      |   6.17 |   6.42 |  86.32 |   6.26 |  36.48 |     -3% |     +1% |  +1299% |
| Conv1d      |   1.62 |   1.49 |   3.05 |   0.90 | 147.44 |     +8% |    -44% |    +88% |
| Conv2d      |   5.20 |   5.19 | 410.99 |   2.11 |  43.67 |     +0% |    -59% |  +7797% |
| Gather      |   3.03 |   3.01 |   4.13 |  15.83 |   9.79 |     +0% |   +423% |    +36% |
| LeakyReLU   |   0.36 |   0.36 |   0.90 |   0.44 |   0.93 |      0% |    +19% |   +146% |
| Linear      |   9.36 |   9.29 |  27.06 |  31.34 | 115.10 |     +0% |   +234% |   +189% |
| MatMul      |  10.93 |   9.89 |  35.71 |  21.59 | 754.10 |    +10% |    +97% |   +226% |
| PReLU       |   0.53 |   0.39 |   3.46 |   0.44 |   0.91 |    +36% |    -17% |   +552% |
| ReLU        |   0.41 |   0.37 |   0.73 |   0.43 |   0.92 |    +11% |     +4% |    +79% |
| Scatter     |   0.31 |   0.31 |  28.25 |   2.77 |   2.31 |      0% |   +788% |  +8959% |
| ScatterSum  |   0.04 |   0.03 |   0.02 |    nan |   1.38 |     +3% |    nan% |    -50% |
| ScatterMax  |   0.04 |   0.03 |   0.02 |    nan |   1.38 |     +7% |    nan% |    -49% |
| SeLU        |   0.49 |   0.43 |   4.85 |   0.51 |   2.66 |    +13% |     +4% |   +899% |
| Sigmoid     |   0.37 |   0.37 |   4.33 |   0.48 |   2.23 |     +1% |    +26% |  +1055% |
| Softmax     |   9.25 |   6.99 |  39.72 |   4.88 |  25.00 |    +32% |    -47% |   +329% |
| Softplus    |   0.41 |   0.37 |  33.75 |   0.47 |   4.73 |     +9% |    +16% |  +8220% |
| Sort        |   1.48 |   1.49 | 242.55 |  22.40 |  51.73 |      0% |  +1414% | +16295% |
| Sum         |   3.24 |   3.22 |   9.11 |   3.09 |  10.11 |     +0% |     -4% |   +180% |
| SumAll      |   2.37 |   2.37 |   6.58 |   2.36 |   3.31 |     +0% |      0% |   +176% |

**M2 Max (cores: 4E+8P+38GPU)** mlx 0.5.0 torch 2.2.1

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.50 |   1.51 |  10.10 |   0.68 |   8.63 |      0% |    -54% |   +571% |
| BCE         |   1.00 |   0.44 |  59.91 |   0.60 |   9.01 |   +126% |    -40% |  +5880% |
| Concat      |   3.18 |   3.16 |  83.04 |   3.34 |  27.24 |     +0% |     +4% |  +2507% |
| Conv1d      |   0.86 |   0.76 |   3.03 |   0.53 | 160.40 |    +13% |    -38% |   +250% |
| Conv2d      |   2.45 |   2.44 | 424.52 |   1.15 |  34.30 |     +0% |    -53% | +17249% |
| Gather      |   1.34 |   1.57 |   3.92 |   8.12 |   8.98 |    -14% |   +504% |   +191% |
| LeakyReLU   |   0.22 |   0.30 |   0.72 |   0.30 |   1.21 |    -24% |    +35% |   +219% |
| Linear      |   5.51 |   5.63 |  23.52 |  12.97 |  37.92 |     -2% |   +135% |   +327% |
| MatMul      |   3.77 |   3.83 |  27.42 |   9.78 |  83.55 |     -1% |   +159% |   +627% |
| PReLU       |   0.28 |   0.48 |   3.27 |   0.42 |   1.03 |    -41% |    +50% |  +1062% |
| ReLU        |   0.37 |   0.24 |   0.62 |   0.35 |   0.94 |    +51% |     -5% |    +67% |
| Scatter     |   0.22 |   0.24 |  28.88 |   1.47 |   1.82 |     -9% |   +567% | +12984% |
| ScatterSum  |   0.03 |   0.03 |   0.01 |    nan |   1.37 |    +10% |    nan% |    -69% |
| ScatterMax  |   0.03 |   0.03 |   0.01 |    nan |   1.39 |    +10% |    nan% |    -68% |
| SeLU        |   0.29 |   0.36 |   4.62 |   0.49 |   7.07 |    -20% |    +69% |  +1511% |
| Sigmoid     |   0.24 |   0.27 |   4.34 |   0.36 |   6.35 |    -10% |    +52% |  +1714% |
| Softmax     |   4.62 |   3.60 |  40.03 |   3.07 |  33.25 |    +28% |    -33% |   +766% |
| Softplus    |   0.25 |   0.24 |  34.73 |   0.33 |   9.39 |     +3% |    +31% | +13696% |
| Sort        |   0.73 |   0.75 | 248.89 |  10.65 |  58.88 |     -2% |  +1360% | +34026% |
| Sum         |   1.61 |   1.64 |   9.22 |   1.96 |  12.05 |     -1% |    +21% |   +472% |
| SumAll      |   1.20 |   1.23 |   6.86 |   1.32 |   3.84 |     -2% |     +9% |   +471% |

**M2 Ultra (cores: 8E+16P+76GPU)** mlx 0.7.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.60 |   1.63 |   9.46 |   0.65 |   9.49 |     -1% |    -59% |   +492% |
| BCE         |   0.64 |   0.45 |  56.57 |   0.47 |   4.23 |    +42% |    -27% |  +8702% |
| Concat      |   1.69 |   1.69 |  81.95 |   1.66 |  38.93 |     +0% |     -1% |  +4743% |
| Conv1d      |   0.55 |   0.51 |   2.64 |   0.45 | 187.91 |     +7% |    -17% |   +382% |
| Conv2d      |   1.35 |   1.38 | 409.78 |   0.67 |  46.05 |     -1% |    -50% | +30276% |
| Gather      |   0.77 |   0.79 |   3.83 |   3.92 |  11.82 |     -2% |   +407% |   +395% |
| LeakyReLU   |   0.32 |   0.25 |   0.85 |   0.21 |   1.99 |    +28% |    -34% |   +162% |
| Linear      |   2.26 |   2.23 |  16.83 |   6.67 |  39.12 |     +1% |   +195% |   +645% |
| MatMul      |   2.53 |   2.53 |  19.21 |   5.59 |  66.55 |      0% |   +121% |   +660% |
| PReLU       |   0.37 |   0.45 |   3.15 |   0.32 |   1.61 |    -18% |    -13% |   +759% |
| ReLU        |   0.29 |   0.24 |   0.67 |   0.33 |   1.61 |    +20% |    +13% |   +132% |
| Scatter     |   0.25 |   0.25 |  27.04 |   0.73 |   1.49 |     +0% |   +193% | +10802% |
| ScatterSum  |   0.03 |   0.03 |   0.01 |    nan |   1.36 |     -1% |    nan% |    -76% |
| ScatterMax  |   0.03 |   0.03 |   0.01 |    nan |   1.37 |    +10% |    nan% |    -76% |
| SeLU        |   0.46 |   0.28 |   4.50 |   0.29 |   1.86 |    +65% |    -36% |   +877% |
| Sigmoid     |   0.24 |   0.25 |   4.11 |   0.26 |   1.71 |     -2% |     +6% |  +1606% |
| Softmax     |   2.47 |   1.88 |  39.27 |   1.35 |  17.90 |    +31% |    -45% |  +1488% |
| Softplus    |   0.27 |   0.26 |  32.13 |   0.26 |   3.53 |     +7% |     -6% | +11598% |
| Sort        |   0.48 |   0.49 | 229.84 |   6.41 |  33.91 |     -1% |  +1231% | +47639% |
| Sum         |   0.90 |   0.91 |   9.22 |   0.95 |   6.80 |     -1% |     +6% |   +925% |
| SumAll      |   0.70 |   0.71 |   6.70 |   0.83 |   1.97 |     -1% |    +19% |   +859% |

**M3 (RAM: 16GB)** - mlx 0.2.0

 Average benchmark:
| Operation      | mlx_gpu | mlx_cpu | mps | cpu | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|----------------|-------|-------|------|------|-------------------|-----------------------|
| Argmax     |   1.20 |  11.63 |   1.71 |   7.10 |    +43% |   +870% |
| BCE        |   4.05 |  40.80 |   8.59 |   8.14 |   +111% |   +906% |
| Concat     |  12.52 |  83.29 |  12.60 |  35.29 |     +0% |   +565% |
| Conv1d     |   2.34 |   3.66 |   1.98 |  71.23 |    -15% |    +56% |
| Conv2d     |  16.47 | 340.03 |   4.43 |  36.36 |    -73% |  +1965% |
| LeakyReLU  |   1.43 |   3.05 |   1.01 |   1.07 |    -29% |   +113% |
| Linear     |  21.55 |  71.89 |  15.84 | 122.32 |    -26% |   +233% |
| MatMul     |  15.49 |  76.57 |  33.24 | 490.48 |   +114% |   +394% |
| PReLU      |   2.36 |   2.76 |   0.99 |   1.11 |    -58% |    +16% |
| ReLU       |   0.76 |   1.39 |   0.96 |   1.01 |    +26% |    +81% |
| SeLU       |   5.23 |   7.72 |   1.02 |   6.88 |    -80% |    +47% |
| Sigmoid    |   0.79 |  26.97 |   1.07 |   5.69 |    +35% |  +3309% |
| Softmax    |   6.31 |  41.35 |  12.08 |  32.54 |    +91% |   +555% |
| Softplus   |   0.73 |  26.82 |   1.08 |   9.09 |    +47% |  +3569% |
| Sort       |  12.67 | 724.26 |  30.73 |  60.29 |   +142% |  +5616% |
| Sum        |   6.96 |  11.24 |   6.61 |  12.27 |     -5% |    +61% |
| SumAll     |   4.26 |   7.79 |   4.78 |   4.38 |    +12% |    +82% |

**M3 Pro (cores: 6E+5P+14GPU)**

| Operation      | mlx_gpu | mlx_cpu | mps | cpu | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|----------------|-------|-------|------|------|-------------------|-----------------------|
| Argmax     |   0.98 |  11.21 |   1.24 |   6.14 |    +25% |  +1041% |
| BCE        |   2.70 |  39.88 |   6.87 |   6.78 |   +154% |  +1374% |
| Concat     |   8.25 |  78.33 |   8.87 |  38.10 |     +7% |   +849% |
| Conv1d     |   2.15 |   3.36 |   2.07 |  83.18 |     -3% |    +56% |
| Conv2d     |  12.06 | 333.03 |   3.09 |  33.87 |    -74% |  +2660% |
| LeakyReLU  |   1.54 |   1.53 |   1.26 |   0.96 |    -18% |      0% |
| Linear     |  15.30 |  52.78 |  11.44 |  91.49 |    -25% |   +244% |
| MatMul     |  16.04 |  69.27 |  22.53 | 390.04 |    +40% |   +331% |
| PReLU      |   2.04 |   2.80 |   1.35 |   0.91 |    -34% |    +37% |
| ReLU       |   0.94 |   0.61 |   1.37 |   0.92 |    +45% |    -34% |
| SeLU       |   3.98 |  10.10 |   1.27 |   4.69 |    -68% |   +153% |
| Sigmoid    |   1.03 |  26.28 |   1.30 |   4.28 |    +25% |  +2446% |
| Softmax    |   4.62 |  32.54 |   9.32 |  29.78 |   +101% |   +604% |
| Softplus   |   1.02 |  25.95 |   1.26 |   6.52 |    +23% |  +2444% |
| Sort       |   8.67 | 711.98 |  21.37 |  46.71 |   +146% |  +8114% |
| Sum        |   4.73 |   9.81 |   5.12 |   8.83 |     +8% |   +107% |
| SumAll     |   3.17 |   4.71 |   3.69 |   3.44 |    +16% |    +48% |

**M3 Max (cores: 4E+12P+40GPU)** mlx 0.2.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.57 |   1.56 |   8.34 |   1.02 |   6.14 |     +0% |    -35% |   +430% |
| BCE         |   1.12 |   0.52 |  38.72 |   0.59 |   3.73 |   +114% |    -47% |  +3362% |
| Concat      |   3.32 |   3.30 |  82.26 |   3.40 |  22.89 |     +0% |     +2% |  +2380% |
| Conv1d      |   0.85 |   0.75 |   2.40 |   0.92 | 156.00 |    +13% |     +8% |   +182% |
| Conv2d      |   4.21 |   4.14 | 329.47 |   1.42 |  31.25 |     +1% |    -66% |  +7723% |
| Gather      |   1.56 |   1.47 |   4.37 |   8.23 |   6.68 |     +5% |   +428% |   +180% |
| LeakyReLU   |   0.43 |   0.29 |   2.57 |   0.54 |   0.66 |    +48% |    +24% |   +491% |
| Linear      |   5.66 |   5.66 |  24.67 |   4.24 |  59.04 |     +0% |    -25% |   +336% |
| MatMul      |   4.20 |   4.19 |  25.57 |   7.62 | 585.74 |     +0% |    +81% |   +508% |
| PReLU       |   0.70 |   0.29 |   2.06 |   0.49 |   0.61 |   +144% |    -29% |   +193% |
| ReLU        |   0.51 |   0.35 |   0.65 |   0.78 |   0.62 |    +45% |    +53% |    +28% |
| Scatter     |   2.29 |   2.22 |  25.40 |   1.66 |   0.93 |     +3% |    -27% |  +1009% |
| ScatterSum  |   0.04 |   0.03 |   0.01 |    nan |   1.22 |    +52% |    nan% |    -81% |
| ScatterMax  |   0.04 |   0.03 |   0.01 |    nan |   1.23 |    +52% |    nan% |    -81% |
| SeLU        |   1.35 |   0.29 |   5.14 |   0.48 |   2.93 |   +361% |    -64% |   +281% |
| Sigmoid     |   0.30 |   0.29 |  26.28 |   0.49 |   2.85 |     +4% |    +62% |  +8629% |
| Softmax     |   4.75 |   3.59 |  35.79 |   3.40 |  16.50 |    +32% |    -28% |   +653% |
| Softplus    |   0.35 |   0.29 |  26.02 |   0.51 |   4.00 |    +21% |    +43% |  +7257% |
| Sort        |   0.77 |   0.76 | 229.39 |   8.04 |  32.43 |     +1% |   +942% | +29646% |
| Sum         |   1.55 |   1.54 |   6.53 |   1.90 |   6.99 |     +0% |    +22% |   +322% |
| SumAll      |   1.19 |   1.19 |   4.78 |   1.32 |   3.22 |     +0% |    +10% |   +300% |

**M4 (6E+4P+10GPU+16GB)** mlx: 0.20.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.56 |   1.49 |   8.33 |   1.43 |   5.39 |     +4% |     -7% |   +434% |
| BCE         |   3.73 |   1.61 |  35.59 |   1.19 |   8.15 |   +131% |    -68% |   +853% |
| Concat      |  12.61 |  12.42 |  50.48 |  12.59 |  29.69 |     +1% |      0% |   +300% |
| Conv1d      |   1.77 |   1.73 |   4.55 |   1.16 |  58.55 |     +2% |    -34% |   +156% |
| Conv2d      |   4.94 |   4.99 |  42.63 |   1.48 |  25.15 |     -1% |    -70% |   +763% |
| Gather      |   3.57 |   3.53 |   3.24 |  34.09 |   9.04 |     +1% |   +854% |     -9% |
| LeakyReLU   |   0.76 |   0.76 |   0.69 |   0.82 |   0.83 |     +0% |     +8% |     -9% |
| Linear      |  12.62 |  12.67 |  60.38 |  13.17 | 116.89 |      0% |     +4% |   +378% |
| MatMul      |  18.27 |  17.17 |  42.77 |  32.16 | 133.45 |     +6% |    +75% |   +134% |
| PReLU       |   0.91 |   0.90 |   2.15 |   0.82 |   0.79 |     +1% |     -9% |   +136% |
| ReLU        |   0.78 |   0.74 |   0.54 |   0.75 |   1.33 |     +5% |     -3% |    -29% |
| Scatter     |   0.82 |   0.79 |   9.34 |   5.89 |   0.98 |     +3% |   +621% |  +1043% |
| ScatterSum  |   0.00 |   0.00 |   0.00 |    nan |   1.08 |    +27% |    nan% |     -7% |
| ScatterMax  |   0.00 |   0.00 |   0.00 |    nan |   1.14 |    +36% |    nan% |     -5% |
| SeLU        |   0.89 |   0.88 |   3.65 |   0.81 |   1.65 |     +1% |     -8% |   +308% |
| Sigmoid     |   0.75 |   0.75 |   3.48 |   0.81 |   1.42 |     +0% |     +7% |   +364% |
| Softmax     |  18.11 |  13.82 |  38.51 |   6.02 |  28.30 |    +31% |    -66% |   +112% |
| Softplus    |   0.83 |   0.76 |  21.28 |   0.78 |   3.51 |     +9% |     -6% |  +2464% |
| Sort        |   1.99 |   1.99 | 218.30 |  32.71 |  98.28 |      0% |  +1545% | +10884% |
| Sum         |   5.90 |   6.18 |   9.00 |   6.70 |  12.98 |     -4% |    +13% |    +52% |
| SumAll      |   4.32 |   4.56 |   6.58 |   4.84 |   5.41 |     -5% |    +12% |    +52% |

**M4 Pro (4E+8P+16GPU+24GB)** mlx: 0.20.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.48 |   1.43 |   7.96 |   1.02 |   5.15 |     +3% |    -30% |   +437% |
| BCE         |   1.47 |   0.70 |  34.00 |   0.70 |   4.26 |   +110% |    -52% |  +2208% |
| Concat      |   5.59 |   5.33 |  48.85 |   5.03 |  27.93 |     +5% |    -10% |   +773% |
| Conv1d      |   1.04 |   1.00 |   4.24 |   0.66 |  85.48 |     +3% |    -36% |   +307% |
| Conv2d      |   3.05 |   3.08 |  32.51 |   0.80 |  29.57 |      0% |    -73% |   +967% |
| Gather      |   2.28 |   2.23 |   3.18 |  13.48 |   7.39 |     +2% |   +491% |    +39% |
| LeakyReLU   |   0.30 |   0.30 |   0.64 |   0.35 |   0.79 |     +0% |    +17% |   +112% |
| Linear      |   7.61 |   7.56 |  40.24 |   7.45 |  63.95 |     +0% |     -2% |   +428% |
| MatMul      |   8.24 |   7.54 |  21.94 |  13.81 | 137.11 |     +9% |    +67% |   +166% |
| PReLU       |   0.43 |   0.46 |   2.13 |   0.37 |   0.84 |     -6% |    -14% |   +394% |
| ReLU        |   0.29 |   0.33 |   0.42 |   0.36 |   1.17 |    -12% |    +25% |    +45% |
| Scatter     |   0.52 |   0.51 |   9.16 |   2.29 |   0.83 |     +2% |   +339% |  +1658% |
| ScatterSum  |   0.00 |   0.00 |   0.00 |    nan |   1.06 |    +34% |    nan% |     -3% |
| ScatterMax  |   0.00 |   0.00 |   0.00 |    nan |   1.03 |    +19% |    nan% |     -5% |
| SeLU        |   0.46 |   0.43 |   3.62 |   0.40 |   1.13 |     +7% |    -11% |   +693% |
| Sigmoid     |   0.28 |   0.29 |   3.46 |   0.34 |   1.01 |     -5% |    +23% |  +1150% |
| Softmax     |   7.23 |   5.56 |  30.51 |   3.08 |  18.53 |    +30% |    -57% |   +321% |
| Softplus    |   0.35 |   0.32 |  21.29 |   0.34 |   2.28 |    +11% |     -4% |  +5944% |
| Sort        |   1.26 |   1.23 | 214.41 |  15.29 |  56.47 |     +2% |  +1112% | +16912% |
| Sum         |   2.25 |   2.28 |   5.91 |   2.61 |   6.09 |     -1% |    +16% |   +163% |
| SumAll      |   1.69 |   1.70 |   4.28 |   1.77 |   1.82 |      0% |     +4% |   +153% |

**M4 Pro (4E+10P+20GPU+24GB)** mlx: 0.24.1

| Operation                      | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|--------------------------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax                     |   1.42 |   1.45 |   7.67 |   0.78 |   5.01 |     -1% |    -45% |   +438% |
| BCE                        |   1.46 |   0.65 |  14.54 |   0.44 |   3.65 |   +124% |    -69% |   +896% |
| Concat                     |   5.54 |   5.22 |  48.61 |   4.96 |  29.00 |     +6% |    -10% |   +777% |
| Conv1d                     |   0.82 |   0.82 |   3.82 |   0.47 | 113.39 |      0% |    -42% |   +364% |
| Conv2d                     |   2.52 |   2.54 |  31.65 |   0.67 |  30.70 |      0% |    -73% |  +1154% |
| Gather                     |   1.30 |   1.32 |   3.13 |  13.28 |   6.85 |     -1% |   +919% |   +140% |
| LayerNorm                  |   0.42 |   0.43 |   2.62 |   0.84 |   1.16 |     -1% |    +96% |   +517% |
| LeakyReLU                  |   0.43 |   0.33 |   0.60 |   0.31 |   0.63 |    +30% |    -27% |    +39% |
| Linear                     |   6.37 |   6.22 |  39.72 |   6.20 |  36.36 |     +2% |     -2% |   +523% |
| MatMul                     |   9.81 |   6.86 |  22.04 |  13.28 |  67.90 |    +43% |    +35% |   +124% |
| PReLU                      |   0.57 |   0.31 |   3.02 |   0.34 |   0.62 |    +83% |    -39% |   +431% |
| ReLU                       |   0.31 |   0.30 |   0.37 |   0.41 |   0.78 |     +1% |    +33% |    +21% |
| ScaledDotProductAttention  |   2.62 |   2.59 |  10.11 |   1.81 |   5.38 |     +1% |    -30% |   +285% |
| Scatter                    |   0.37 |   0.32 |   9.06 |   2.29 |   0.75 |    +17% |   +512% |  +2320% |
| ScatterSum                 |   0.00 |   0.00 |   0.00 |   0.28 |   0.97 |    +18% | +22629% |     +2% |
| ScatterMax                 |   0.00 |   0.00 |   0.00 |   0.29 |   0.98 |    +33% | +24052% |     +0% |
| SeLU                       |   0.95 |   0.34 |   6.10 |   0.32 |   0.93 |   +176% |    -66% |   +541% |
| Sigmoid                    |   0.36 |   0.36 |   2.19 |   0.28 |   0.84 |     +1% |    -22% |   +504% |
| Softmax                    |   7.17 |   5.37 |  32.41 |   2.77 |  13.97 |    +33% |    -61% |   +352% |
| Softplus                   |   0.32 |   0.32 |  20.25 |   0.30 |   1.86 |     -1% |     -6% |  +6274% |
| Sort                       |   1.29 |   1.24 | 209.96 |  14.67 |  45.28 |     +4% |  +1039% | +16202% |
| Sum                        |   2.25 |   2.21 |   9.85 |   2.49 |   5.12 |     +1% |    +10% |   +338% |
| SumAll                     |   1.67 |   1.68 |   7.18 |   1.71 |   1.62 |      0% |     +2% |   +329% |

**M4 Max (4E+12P+40GPU+128GB)** mlx: 0.20.0

| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.41 |   1.43 |   8.01 |   0.70 |   4.69 |     -1% |    -50% |   +468% |
| BCE         |   0.88 |   0.89 |  33.88 |   0.52 |   2.93 |     -1% |    -40% |  +3770% |
| Concat      |   2.86 |   2.87 |  47.64 |   2.67 |  19.94 |      0% |     -6% |  +1563% |
| Conv1d      |   0.59 |   0.51 |   3.81 |   0.40 | 110.98 |    +15% |    -31% |   +542% |
| Conv2d      |   1.43 |   1.43 |  32.19 |   0.60 |  26.71 |     +0% |    -58% |  +2152% |
| Gather      |   1.08 |   1.04 |   3.07 |   7.37 |   6.20 |     +3% |   +584% |   +185% |
| LeakyReLU   |   0.24 |   0.23 |   0.67 |   0.19 |   0.75 |     +2% |    -17% |   +181% |
| Linear      |   3.35 |   3.83 |  39.12 |   3.21 |  45.63 |    -12% |     -4% |  +1069% |
| MatMul      |   4.12 |   4.21 |  21.00 |   6.36 | 139.36 |     -2% |    +54% |   +409% |
| PReLU       |   0.38 |   0.29 |   2.10 |   0.28 |   0.87 |    +30% |    -26% |   +445% |
| ReLU        |   0.23 |   0.22 |   0.42 |   0.27 |   0.87 |     +3% |    +21% |    +86% |
| Scatter     |   0.28 |   0.27 |   9.08 |   1.25 |   0.65 |     +2% |   +343% |  +3122% |
| ScatterSum  |   0.00 |   0.00 |   0.00 |    nan |   1.06 |    +28% |    nan% |     -7% |
| ScatterMax  |   0.00 |   0.00 |   0.00 |    nan |   1.06 |    +10% |    nan% |     -3% |
| SeLU        |   0.34 |   0.29 |   3.63 |   0.35 |   1.12 |    +18% |     +2% |   +967% |
| Sigmoid     |   0.21 |   0.22 |   3.45 |   0.27 |   0.93 |     -6% |    +27% |  +1551% |
| Softmax     |   3.97 |   3.40 |  29.65 |   1.52 |  13.74 |    +16% |    -61% |   +647% |
| Softplus    |   0.29 |   0.26 |  21.32 |   0.25 |   1.88 |    +11% |    -13% |  +7339% |
| Sort        |   0.58 |   0.59 | 207.83 |   7.18 |  41.13 |      0% |  +1128% | +35475% |
| Sum         |   1.25 |   1.23 |   6.28 |   1.44 |   4.54 |     +1% |    +15% |   +403% |
| SumAll      |   0.95 |   0.93 |   4.61 |   1.07 |   1.48 |     +1% |    +13% |   +387% |

**M3 Ultra (8E+20P+60GPU+96GB)** mlx: 0.24.1
| Operation       | mlx_gpu | mlx_gpu_compile | mlx_cpu | mps | cpu | mlx_gpu_compile/mlx_gpu speedup | mlx_gpu/mps speedup | mlx_gpu/mlx_cpu speedup |
|-----------------|-------|---------------|-------|------|------|-------------------------------|-------------------|-----------------------|
| Argmax      |   1.77 |   1.70 |   8.40 |   0.59 |   7.99 |     +4% |    -66% |   +373% |
| BCE         |   0.68 |   0.37 |  15.70 |   0.54 |   2.77 |    +82% |    -21% |  +2201% |
| Concat      |   1.88 |   1.85 |  84.41 |   1.80 |  40.87 |     +1% |     -4% |  +4379% |
| Conv1d      |   0.52 |   0.52 |   4.16 |   0.44 | 148.64 |     +0% |    -14% |   +700% |
| Conv2d      |   1.32 |   1.30 |  31.36 |   0.55 |  33.67 |     +1% |    -58% |  +2272% |
| Gather      |   0.60 |   0.72 |   3.38 |   4.11 |  14.71 |    -15% |   +579% |   +458% |
| LeakyReLU   |   0.26 |   0.26 |   0.67 |   0.22 |   1.47 |      0% |    -13% |   +158% |
| Linear      |   2.74 |   2.77 |  52.00 |   2.53 |  45.65 |     -1% |     -7% |  +1800% |
| MatMul      |   3.38 |   3.54 |  16.76 |   4.96 | 231.43 |     -4% |    +46% |   +395% |
| PReLU       |   0.62 |   0.48 |   3.15 |   0.26 |   1.67 |    +29% |    -58% |   +412% |
| ReLU        |   0.26 |   0.35 |   0.42 |   0.22 |   1.47 |    -25% |    -12% |    +63% |
| Scatter     |   0.26 |   0.26 |  10.27 |   0.82 |   0.93 |     +2% |   +211% |  +3792% |
| ScatterSum  |   0.00 |   0.00 |   0.00 |   0.41 |   1.25 |    +43% | +18803% |    -33% |
| ScatterMax  |   0.00 |   0.00 |   0.00 |   1.22 |   1.23 |    +53% | +60915% |    -27% |
| SeLU        |   0.60 |   0.43 |   6.39 |   0.25 |   1.29 |    +40% |    -58% |   +963% |
| Sigmoid     |   0.26 |   0.26 |   2.21 |   0.27 |   1.25 |     +2% |     +2% |   +740% |
| Softmax     |   2.67 |   2.08 |  33.54 |   1.22 |  11.51 |    +28% |    -54% |  +1155% |
| Softplus    |   0.29 |   0.25 |  24.63 |   0.23 |   1.76 |    +14% |    -19% |  +8357% |
| Sort        |   0.73 |   0.60 | 213.11 |   6.18 |  33.69 |    +22% |   +746% | +29113% |
| Sum         |   0.96 |   0.96 |  10.85 |   1.09 |   4.35 |      0% |    +13% |  +1029% |
| SumAll      |   0.73 |   0.76 |   7.93 |   0.84 |   1.67 |     -3% |    +15% |   +982% |

## CUDA GPUs

**Tesla V100 PCIe (32Go / Intel Xeon Gold 5120 14 cores / 28 threads @ 2.2GHz (Skylake), 60Go)**

| Operation       | cpu | cuda | cuda/cpu speedup |
|-----------------|------|------|----------------|
| Argmax      |  34.34 |   0.10 | +33411% |
| BCE         | 198.19 |   0.19 | +102820% |
| Concat      | 380.98 |   1.67 | +22679% |
| Conv1d      |  30.21 |   0.33 |  +9027% |
| Conv2d      |  52.73 |   0.87 |  +5938% |
| Gather      |  96.61 |   0.42 | +22636% |
| LeakyReLU   |   5.51 |   0.08 |  +7010% |
| Linear      | 901.98 |   3.79 | +23722% |
| MatMul      | 1241.12 |   2.80 | +44293% |
| PReLU       |   5.55 |   0.08 |  +7159% |
| ReLU        |   5.50 |   0.08 |  +7032% |
| Scatter     |   6.92 |   0.12 |  +5875% |
| ScatterSum  |   4.25 |   0.08 |  +5058% |
| ScatterMax  |    nan |    nan |    nan% |
| SeLU        |  11.56 |   0.08 | +14709% |
| Sigmoid     |   9.46 |   0.08 | +12023% |
| Softmax     | 221.43 |   0.71 | +31300% |
| Softplus    |  22.13 |   0.08 | +27658% |
| Sort        | 526.33 |   2.59 | +20202% |
| Sum         |  67.43 |   0.70 |  +9472% |
| SumAll      |  29.82 |   0.50 |  +5822% |

**Tesla V100 NVLink (32Go / Intel Xeon Gold 6148 20 cores, 40 threads @ 2.4 GHz (Skylake), 60Go)**

| Operation       | cpu | cuda | cuda/cpu speedup |
|-----------------|------|------|----------------|
| Argmax      |  28.23 |   0.10 | +28460% |
| BCE         | 186.05 |   0.19 | +97956% |
| Concat      | 531.34 |   1.67 | +31744% |
| Conv1d      |  22.37 |   0.31 |  +7033% |
| Conv2d      |  52.89 |   0.83 |  +6257% |
| Gather      | 161.56 |   0.41 | +39152% |
| LeakyReLU   |  16.95 |   0.08 | +21591% |
| Linear      | 666.79 |   3.58 | +18532% |
| MatMul      | 998.29 |   2.68 | +37198% |
| PReLU       |  15.55 |   0.08 | +20584% |
| ReLU        |  14.07 |   0.08 | +18496% |
| Scatter     |   6.19 |   0.11 |  +5548% |
| ScatterSum  |   6.83 |   0.08 |  +8757% |
| ScatterMax  |    nan |    nan |    nan% |
| SeLU        |  20.94 |   0.08 | +27171% |
| Sigmoid     |  19.82 |   0.08 | +25331% |
| Softmax     | 253.76 |   0.70 | +36156% |
| Softplus    |  29.21 |   0.08 | +37131% |
| Sort        | 422.98 |   2.48 | +16933% |
| Sum         |  69.38 |   0.70 |  +9861% |
| SumAll      |  31.13 |   0.50 |  +6152% |

**RTX4090 ((Desktop) / 10th Gen Intel Core i9-10940X @ 3.30GHz 128GB)**

| Operation       | cpu | cuda | cuda/cpu speedup |
|-----------------|------|------|----------------|
| Argmax      |   6.67 |   0.04 | +14782% |
| BCE         |  23.74 |   0.14 | +16992% |
| Concat      |  52.08 |   1.29 |  +3922% |
| Conv1d      |   2.84 |   0.15 |  +1753% |
| Conv2d      |   6.60 |   0.25 |  +2559% |
| Gather      |  19.75 |   0.27 |  +7340% |
| LeakyReLU   |   2.44 |   0.03 |  +7439% |
| Linear      |  62.27 |   1.01 |  +6057% |
| MatMul      |  87.47 |   1.36 |  +6322% |
| PReLU       |   2.28 |   0.04 |  +5297% |
| ReLU        |   2.47 |   0.03 |  +7216% |
| Scatter     |   1.84 |   0.07 |  +2652% |
| ScatterSum  |   3.86 |   0.06 |  +5919% |
| ScatterMax  |   3.86 |   0.08 |  +4790% |
| SeLU        |   2.71 |   0.04 |  +6952% |
| Sigmoid     |   2.63 |   0.05 |  +5626% |
| Softmax     |  27.75 |   0.59 |  +4634% |
| Softplus    |   3.50 |   0.04 |  +8149% |
| Sort        |  46.67 |   0.90 |  +5077% |
| Sum         |  12.19 |   0.62 |  +1866% |
| SumAll      |   6.95 |   0.45 |  +1428% |

A100 80GB 80GB PCIe ((Server) / Intel(R) Xeon(R) Gold 6254 CPU @ 3.10GHz, 754GB)

| Operation       | cpu | cuda | cuda/cpu speedup |
|-----------------|------|------|----------------|
| Argmax      |   5.04 |   0.06 |  +7856% |
| BCE         |  18.22 |   0.11 | +16097% |
| Concat      |  30.47 |   0.74 |  +4036% |
| Conv1d      | 1029.44 |   0.13 | +811270% |
| Conv2d      | 531.83 |   0.26 | +205989% |
| Gather      |   9.59 |   0.30 |  +3045% |
| LeakyReLU   |   1.68 |   0.06 |  +2579% |
| Linear      |  47.44 |   2.17 |  +2090% |
| MatMul      |  50.91 |   2.07 |  +2355% |
| PReLU       |   1.60 |   0.05 |  +3332% |
| ReLU        |   1.43 |   0.04 |  +3380% |
| Scatter     |   1.61 |   0.11 |  +1358% |
| ScatterSum  |   4.95 |   0.06 |  +7547% |
| ScatterMax  |   5.39 |   0.33 |  +1511% |
| SeLU        |   1.82 |   0.04 |  +4259% |
| Sigmoid     |   3.03 |   0.04 |  +7553% |
| Softmax     |  18.18 |   0.36 |  +5003% |
| Softplus    |   2.87 |   0.04 |  +6412% |
| Sort        |  52.86 |   1.16 |  +4449% |
| Sum         |  11.38 |   0.37 |  +2947% |
| SumAll      |   6.85 |   0.29 |  +2226% |

