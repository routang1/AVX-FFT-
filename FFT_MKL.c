#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "mkl_dfti.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FFT_SIZE 4096
#define REPEAT 1000
#define OUTPUT_COUNT 8

/* FFTAVX.c 中的现有实现，链接时用于正确性和性能对照。 */
void fft_avx256(float *real, float *imag, int N);

typedef struct {
    DFTI_DESCRIPTOR_HANDLE descriptor;
    LARGE_INTEGER frequency;
} FFT_MKL_PLAN;

static int report_mkl_error(MKL_LONG status, const char *api_name)
{
    if (!DftiErrorClass(status, DFTI_NO_ERROR)) {
        fprintf(stderr, "%s failed: %s\n", api_name, DftiErrorMessage(status));
        return 0;
    }
    return 1;
}

/* 创建单精度复数 FFT，DFTI_REAL_REAL 对应分离的 real[]/imag[]。 */
static int FFT_MKL_Init(FFT_MKL_PLAN *plan, int N)
{
    MKL_LONG status;

    plan->descriptor = NULL;
    status = DftiCreateDescriptor(
        &plan->descriptor,
        DFTI_SINGLE,
        DFTI_COMPLEX,
        1,
        (MKL_LONG)N
    );
    if (!report_mkl_error(status, "DftiCreateDescriptor")) {
        return 0;
    }

    status = DftiSetValue(
        plan->descriptor,
        DFTI_COMPLEX_STORAGE,
        DFTI_REAL_REAL
    );
    if (!report_mkl_error(status, "DftiSetValue")) {
        DftiFreeDescriptor(&plan->descriptor);
        return 0;
    }

    status = DftiCommitDescriptor(plan->descriptor);
    if (!report_mkl_error(status, "DftiCommitDescriptor")) {
        DftiFreeDescriptor(&plan->descriptor);
        return 0;
    }

    if (!QueryPerformanceFrequency(&plan->frequency)) {
        fprintf(stderr, "QueryPerformanceFrequency failed\n");
        DftiFreeDescriptor(&plan->descriptor);
        return 0;
    }

    return 1;
}

/* 计时区间仅包含 DftiComputeForward。 */
int FFT_MKL(
    FFT_MKL_PLAN *plan,
    float *real,
    float *imag,
    double *elapsed_us
)
{
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    MKL_LONG status;

    QueryPerformanceCounter(&start);
    status = DftiComputeForward(plan->descriptor, real, imag);
    QueryPerformanceCounter(&end);

    *elapsed_us =
        (double)(end.QuadPart - start.QuadPart)
        * 1000000.0
        / (double)plan->frequency.QuadPart;

    return report_mkl_error(status, "DftiComputeForward");
}

static int FFT_MKL_Destroy(FFT_MKL_PLAN *plan)
{
    MKL_LONG status = DftiFreeDescriptor(&plan->descriptor);
    return report_mkl_error(status, "DftiFreeDescriptor");
}

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

int main(void)
{
    const int N = FFT_SIZE;
    const int repeat = REPEAT;
    _Alignas(32) float real[FFT_SIZE];
    _Alignas(32) float imag[FFT_SIZE];
    _Alignas(32) float mkl_real[FFT_SIZE];
    _Alignas(32) float mkl_imag[FFT_SIZE];
    _Alignas(32) float avx_real[FFT_SIZE];
    _Alignas(32) float avx_imag[FFT_SIZE];
    FFT_MKL_PLAN plan;
    LARGE_INTEGER avx_start;
    LARGE_INTEGER avx_end;
    double mkl_total_us = 0.0;
    double avx_total_us = 0.0;
    double compute_us;

    init_input(real, imag, N);

    if (!FFT_MKL_Init(&plan, N)) {
        return 1;
    }

    /* 不计时预热，避免首次调用的惰性初始化影响 benchmark。 */
    memcpy(mkl_real, real, sizeof(real));
    memcpy(mkl_imag, imag, sizeof(imag));
    if (!FFT_MKL(&plan, mkl_real, mkl_imag, &compute_us)) {
        FFT_MKL_Destroy(&plan);
        return 1;
    }

    for (int r = 0; r < repeat; r++) {
        memcpy(mkl_real, real, sizeof(real));
        memcpy(mkl_imag, imag, sizeof(imag));

        if (!FFT_MKL(&plan, mkl_real, mkl_imag, &compute_us)) {
            FFT_MKL_Destroy(&plan);
            return 1;
        }
        mkl_total_us += compute_us;
    }

    for (int r = 0; r < repeat; r++) {
        memcpy(avx_real, real, sizeof(real));
        memcpy(avx_imag, imag, sizeof(imag));

        QueryPerformanceCounter(&avx_start);
        fft_avx256(avx_real, avx_imag, N);
        QueryPerformanceCounter(&avx_end);

        avx_total_us += get_elapsed_us(avx_start, avx_end, plan.frequency);
    }

    printf("\nMKL FFT OUT (first %d bins)\n", OUTPUT_COUNT);
    for (int k = 0; k < OUTPUT_COUNT; k++) {
        printf("X[%d] = %f %+.6fi\n", k, mkl_real[k], mkl_imag[k]);
    }

    printf("\nN = %d, repeat = %d\n", N, repeat);
    printf("MKL DftiComputeForward average time: %.6f us\n",
           mkl_total_us / (double)repeat);
    printf("AVX2 fft_avx256 average time:       %.6f us\n",
           avx_total_us / (double)repeat);
    printf("max error: %.9e\n",
           max_complex_error(
               mkl_real,
               mkl_imag,
               avx_real,
               avx_imag,
               N
           ));

    if (!FFT_MKL_Destroy(&plan)) {
        return 1;
    }

    return 0;
}
