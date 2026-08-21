# AVX256 FFT

使用 AVX256 实现单精度复数 FFT，并与 Intel oneMKL 比较性能和误差。

## 目录

```text
src/        FFT 实现和头文件
benchmark/  统一测试入口
```

## 编译

```powershell
cd D:\Work_project\DSA\Code\C

& "D:\MSYS\ucrt64\bin\gcc.exe" -std=c17 -O2 -Wall -Wextra -mavx2 -mfma -ffp-contract=off benchmark\benchmark.c src\fft_avx256.c src\fft_mkl.c -Isrc -I "D:\Work_project\DSA\oneMKL\mkl\latest\include" "D:\Work_project\DSA\oneMKL\mkl\latest\lib\mkl_rt.lib" -lm -o benchmark\benchmark.exe

```

## 运行

```powershell

cmd.exe /d /s /c 'call "D:\Work_project\DSA\oneMKL\setvars.bat" && "D:\Work_project\DSA\Code\C\benchmark\benchmark.exe"'

```
当前测试使用 4096 点 FFT，重复 1000 次，并通过 `QueryPerformanceCounter` 计时。

通用 `m >= 16` 路径已使用 FMA 复数乘法。10 次测试均值从 `12.328940 us` 降至 `11.452840 us`，加速比为 `1.0765x`。

## 测试情况

- 1.0 原始AVXFFT结果：

N = 4096, repeat = 1000
MKL DftiComputeForward average time: 6.854700 us
AVX2 fft_avx256 average time:       12.733800 us
max error: 2.321016564e-02

- 1.1 通用路径使用 FMA 后（10 次测试均值）：

AVX2 fft_avx256 average time:       11.452840 us
max error: 2.308916171e-02
speedup: 1.0765x

- 1.2 `m == 4` 使用特殊旋转因子后（8 组交替配对均值）：

AVX2 before: 11.514450 us
AVX2 after:  11.400737 us
max error:   2.308916171e-02
speedup:     1.009974x

本次端到端提升约 1%，接近系统测试噪声，暂不能视为显著性能提升。
