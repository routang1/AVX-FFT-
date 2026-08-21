# AVX256 FFT

使用 AVX256 实现单精度复数 FFT，并与 Intel oneMKL 比较性能和误差。

## 目录

```text
src/        FFT 实现和头文件
benchmark/  统一测试入口
```

## 编译

```powershell
D:\MSYS\ucrt64\bin\gcc.exe -std=c17 -O2 -mavx2 benchmark\benchmark.c src\fft_avx256.c src\fft_mkl.c -Isrc -I D:\Work_project\DSA\oneMKL\mkl\latest\include D:\Work_project\DSA\oneMKL\mkl\latest\lib\mkl_rt.lib -lm -o benchmark\benchmark.exe
```

## 运行

```powershell
cmd.exe /d /s /c 'call "D:\Work_project\DSA\oneMKL\setvars.bat" && "D:\Work_project\DSA\Code\C\benchmark\benchmark.exe"'
```

当前测试使用 4096 点 FFT，重复 1000 次，并通过 `QueryPerformanceCounter` 计时。
