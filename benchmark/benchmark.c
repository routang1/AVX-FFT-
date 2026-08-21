#include <math.h>
#include <stdio.h>
#include <string.h>
#include <windows.h>

#include "fft_avx256.h"
#include "fft_mkl.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FFT_SIZE 4096
#define REPEAT 1000
#define OUTPUT_COUNT 8


//计算时间
static double get_elapsed_us(
    LARGE_INTEGER start,
    LARGE_INTEGER end,
    LARGE_INTEGER frequency
)
{
    return
        (double)(end.QuadPart - start.QuadPart)
        * 1000000.0
        / (double)frequency.QuadPart;
}


//生成信号
static void init_input(float *real, float *imag, int N)
{
    for (int i = 0; i < N; i++) {
        float phase = (float)(2.0 * M_PI * i / N);
        real[i] =
            sinf(37.0f * phase)
            + 0.5f * cosf(123.0f * phase)
            + 0.25f * sinf(511.0f * phase);
        imag[i] = 0.0f;
    }
}

//计算误差
static double max_complex_error(
    const float *real_a,
    const float *imag_a,
    const float *real_b,
    const float *imag_b,
    int N
)
{
    double max_error = 0.0;

    for (int i = 0; i < N; i++) {
        double real_error = (double)real_a[i] - (double)real_b[i];
        double imag_error = (double)imag_a[i] - (double)imag_b[i];
        double error = sqrt(real_error * real_error + imag_error * imag_error);

        if (error > max_error) {
            max_error = error;
        }
    }

    return max_error;
}


int main(void)
{
    _Alignas(32) float real[FFT_SIZE];
    _Alignas(32) float imag[FFT_SIZE];
    _Alignas(32) float mkl_real[FFT_SIZE];
    _Alignas(32) float mkl_imag[FFT_SIZE];
    _Alignas(32) float avx_real[FFT_SIZE];
    _Alignas(32) float avx_imag[FFT_SIZE];
    FFT_MKL_PLAN mkl_plan;
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    double mkl_total_us = 0.0;
    double avx_total_us = 0.0;

    init_input(real, imag, FFT_SIZE);

    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "QueryPerformanceFrequency failed\n");
        return 1;
    }
    if (!fft_mkl_init(&mkl_plan, FFT_SIZE)) {
        return 1;
    }

    //两种算法都先执行一次不计时预热
    memcpy(mkl_real, real, sizeof(real));
    memcpy(mkl_imag, imag, sizeof(imag));
    if (!fft_mkl_forward(&mkl_plan, mkl_real, mkl_imag)) {
        fft_mkl_destroy(&mkl_plan);
        return 1;
    }

    memcpy(avx_real, real, sizeof(real));
    memcpy(avx_imag, imag, sizeof(imag));
    fft_avx256(avx_real, avx_imag, FFT_SIZE);
    
    //开始计时
    for (int r = 0; r < REPEAT; r++) {
        memcpy(mkl_real, real, sizeof(real));
        memcpy(mkl_imag, imag, sizeof(imag));

        QueryPerformanceCounter(&start);
        if (!fft_mkl_forward(&mkl_plan, mkl_real, mkl_imag)) {
            fft_mkl_destroy(&mkl_plan);
            return 1;
        }
        QueryPerformanceCounter(&end);
        mkl_total_us += get_elapsed_us(start, end, frequency);
    }

    for (int r = 0; r < REPEAT; r++) {
        memcpy(avx_real, real, sizeof(real));
        memcpy(avx_imag, imag, sizeof(imag));

        QueryPerformanceCounter(&start);
        fft_avx256(avx_real, avx_imag, FFT_SIZE);
        QueryPerformanceCounter(&end);
        avx_total_us += get_elapsed_us(start, end, frequency);
    }

    double mkl_avg_us = mkl_total_us / (double)REPEAT;
    double avx_avg_us = avx_total_us / (double)REPEAT;
    double avx_over_mkl = avx_avg_us / mkl_avg_us;
    double mkl_speedup_vs_avx = avx_avg_us / mkl_avg_us;

    printf("\nMKL FFT OUT (first %d bins)\n", OUTPUT_COUNT);
    for (int k = 0; k < OUTPUT_COUNT; k++) {
        printf("X[%d] = %f %+.6fi\n", k, mkl_real[k], mkl_imag[k]);
    }

    printf("\nN = %d, repeat = %d\n", FFT_SIZE, REPEAT);
    printf("MKL DftiComputeForward average time: %.6f us\n",
           mkl_avg_us);
    printf("AVX2 fft_avx256 average time:       %.6f us\n",
           avx_avg_us);
    printf("AVX / MKL time ratio:               %.4f\n",
           avx_over_mkl);
    printf("MKL speedup vs AVX:                 %.4fx\n",
           mkl_speedup_vs_avx);
    printf("max error: %.9e\n",
           max_complex_error(
               mkl_real,
               mkl_imag,
               avx_real,
               avx_imag,
               FFT_SIZE
           ));

    if (!fft_mkl_destroy(&mkl_plan)) {
        return 1;
    }

    return 0;
}
