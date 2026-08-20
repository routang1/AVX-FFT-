#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define FFT_SIZE 4096
#define REPEAT 1000
#define OUTPUT_COUNT 8

static void swap(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

// 位反转操作
static void bit_reverse(float *real, float *imag, int N) {
    int j = 0;
    for (int i = 0; i < N - 1; i++) {
        if (i < j) {
            swap(&real[i], &real[j]);
            swap(&imag[i], &imag[j]);
        }
        int k = N / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }
}

// 快速傅里叶变换函数
void fft_avx256(float *real, float *imag, int N) 
{
    bit_reverse(real, imag, N);
    int log2N = 0;
    for (int n = N; n > 1; n >>= 1) {
        log2N++;
    }

    for (int s = 1; s <= log2N; s++) {
        int m = 1 << s;
        _Alignas(32) float w_real_vec[8], w_imag_vec[8];
        if(m==2)
        {
            for (int k = 0; k < N; k += 8) {
                __m256 w_real = _mm256_set1_ps(-1.0f);
                __m256 w_imag = _mm256_set1_ps(0.0f);
                __m256 x_real = _mm256_load_ps(real + k);
                //从 real[k] 开始，连续读取 8 个 float，装进一个 256 位 AVX 寄存器 x_real。
                __m256 x_imag = _mm256_load_ps(imag + k);
                __m256 a_real = _mm256_shuffle_ps(x_real,x_real,0b10100000);
                __m256 a_imag = _mm256_shuffle_ps(x_imag,x_imag,0b10100000);
                __m256 b_real = _mm256_shuffle_ps(x_real,x_real,0b11110101);
                __m256 b_imag = _mm256_shuffle_ps(x_imag,x_imag,0b11110101);
                __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(w_real,b_real),_mm256_mul_ps(w_imag,b_imag));
                __m256 r_imag = _mm256_add_ps(_mm256_mul_ps(w_real,b_imag),_mm256_mul_ps(w_imag,b_real));
                __m256 rst_real = _mm256_addsub_ps(a_real,r_real);
                __m256 rst_imag = _mm256_addsub_ps(a_imag,r_imag);
                _mm256_store_ps(real + k,rst_real);
                _mm256_store_ps(imag + k,rst_imag);
            }
        }
        else if(m==4)
        {
            w_real_vec[0] = 1.0f;
            w_real_vec[1] = 6.12323426e-17f;
            w_real_vec[2] = -w_real_vec[0];
            w_real_vec[3] = -w_real_vec[1];
            w_real_vec[4] = w_real_vec[0];
            w_real_vec[5] = w_real_vec[1];
            w_real_vec[6] = -w_real_vec[0];
            w_real_vec[7] = -w_real_vec[1];

            w_imag_vec[0] = 0.0f;
            w_imag_vec[1] = -1.0f;
            w_imag_vec[2] = -w_imag_vec[0];
            w_imag_vec[3] = -w_imag_vec[1];
            w_imag_vec[4] = w_imag_vec[0];
            w_imag_vec[5] = w_imag_vec[1];
            w_imag_vec[6] = -w_imag_vec[0];
            w_imag_vec[7] = -w_imag_vec[1];
            for (int k = 0; k < N; k += 8) {
                __m256 w_real = _mm256_load_ps(w_real_vec);
                __m256 w_imag = _mm256_load_ps(w_imag_vec);
                __m256 x_real = _mm256_load_ps(real + k);
                __m256 x_imag = _mm256_load_ps(imag + k);
                __m256 a_real = _mm256_shuffle_ps(x_real,x_real,0b01000100);
                __m256 a_imag = _mm256_shuffle_ps(x_imag,x_imag,0b01000100);
                __m256 b_real = _mm256_shuffle_ps(x_real,x_real,0b11101110);
                __m256 b_imag = _mm256_shuffle_ps(x_imag,x_imag,0b11101110);
                __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(w_real,b_real),_mm256_mul_ps(w_imag,b_imag));
                __m256 r_imag = _mm256_add_ps(_mm256_mul_ps(w_real,b_imag),_mm256_mul_ps(w_imag,b_real));
                __m256 rst_real = _mm256_add_ps(a_real,r_real);
                __m256 rst_imag = _mm256_add_ps(a_imag,r_imag);
                _mm256_store_ps(real + k,rst_real);
                _mm256_store_ps(imag + k,rst_imag);
            }
        }
        else if(m==8)
        {
            w_real_vec[0] = 1.0f;
            w_imag_vec[0] = 0.0f;
            w_real_vec[1] = 0.707106769f;
            w_imag_vec[1] = -0.707106769f;
            w_real_vec[2] = 0.0f;
            w_imag_vec[2] = -0.99999994f;
            w_real_vec[3] = -0.707106709f;
            w_imag_vec[3] = -0.707106709f;
            w_real_vec[4] = -w_real_vec[0];
            w_imag_vec[4] = -w_imag_vec[0];
            w_real_vec[5] = -w_real_vec[1];
            w_imag_vec[5] = -w_imag_vec[1];
            w_real_vec[6] = -w_real_vec[2];
            w_imag_vec[6] = -w_imag_vec[2];
            w_real_vec[7] = -w_real_vec[3];
            w_imag_vec[7] = -w_imag_vec[3];
            for (int k = 0; k < N; k += 8) {
                __m256 w_real = _mm256_load_ps(w_real_vec);
                __m256 w_imag = _mm256_load_ps(w_imag_vec);
                __m256 x_real = _mm256_load_ps(real + k);
                __m256 x_imag = _mm256_load_ps(imag + k);
                __m256 a_real = _mm256_permute2f128_ps(x_real,x_real,0x00);
                __m256 a_imag = _mm256_permute2f128_ps(x_imag,x_imag,0x00);
                __m256 b_real = _mm256_permute2f128_ps(x_real,x_real,0x11);
                __m256 b_imag = _mm256_permute2f128_ps(x_imag,x_imag,0x11);
                __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(w_real,b_real),_mm256_mul_ps(w_imag,b_imag));
                __m256 r_imag = _mm256_add_ps(_mm256_mul_ps(w_real,b_imag),_mm256_mul_ps(w_imag,b_real));
                __m256 rst_real = _mm256_add_ps(a_real,r_real);
                __m256 rst_imag = _mm256_add_ps(a_imag,r_imag);
                _mm256_store_ps(real + k,rst_real);
                _mm256_store_ps(imag + k,rst_imag);
            }
        }
        else
        {
            float wm_real = cosf((float)(2.0 * M_PI / m));
            float wm_imag = -sinf((float)(2.0 * M_PI / m));

            w_real_vec[0] = 1.0f;
            w_imag_vec[0] = 0.0f;
            for (int i = 1; i < 8; i++) {
                w_real_vec[i] =
                    w_real_vec[i - 1] * wm_real
                    - w_imag_vec[i - 1] * wm_imag;
                w_imag_vec[i] =
                    w_real_vec[i - 1] * wm_imag
                    + w_imag_vec[i - 1] * wm_real;
            }

            float w8_real =
                w_real_vec[7] * wm_real - w_imag_vec[7] * wm_imag;
            float w8_imag =
                w_real_vec[7] * wm_imag + w_imag_vec[7] * wm_real;
            __m256 w8_real_vec = _mm256_set1_ps(w8_real);
            __m256 w8_imag_vec = _mm256_set1_ps(w8_imag);

            for (int k = 0; k < N; k += m) {
                __m256 w_real = _mm256_load_ps(w_real_vec);
                __m256 w_imag = _mm256_load_ps(w_imag_vec);
                for (int j = 0; j < m / 2; j+=8) {
                    __m256 x_real = _mm256_load_ps(real + k + j);
                    __m256 x_imag = _mm256_load_ps(imag + k + j);
                    __m256 x2_real = _mm256_load_ps(real + k + j + m/2);
                    __m256 x2_imag = _mm256_load_ps(imag + k + j + m/2);
                    __m256 r_real = _mm256_sub_ps(_mm256_mul_ps(w_real,x2_real),_mm256_mul_ps(w_imag,x2_imag));
                    __m256 r_imag = _mm256_add_ps(_mm256_mul_ps(w_real,x2_imag),_mm256_mul_ps(w_imag,x2_real));
                    __m256 rst_real = _mm256_add_ps(x_real,r_real);
                    __m256 rst_imag = _mm256_add_ps(x_imag,r_imag);
                    __m256 rst2_real = _mm256_sub_ps(x_real,r_real);
                    __m256 rst2_imag = _mm256_sub_ps(x_imag,r_imag);
                    _mm256_store_ps(real + k + j,rst_real);
                    _mm256_store_ps(imag + k + j,rst_imag);
                    _mm256_store_ps(real + k + j + m/2,rst2_real);
                    _mm256_store_ps(imag + k + j + m/2,rst2_imag);

                    __m256 next_w_real = _mm256_sub_ps(
                        _mm256_mul_ps(w_real, w8_real_vec),
                        _mm256_mul_ps(w_imag, w8_imag_vec)
                    );
                    w_imag = _mm256_add_ps(
                        _mm256_mul_ps(w_real, w8_imag_vec),
                        _mm256_mul_ps(w_imag, w8_real_vec)
                    );
                    w_real = next_w_real;
                }
            }
        }
    }
}

void fft(float *real, float *imag, int N) {
    bit_reverse(real, imag, N);

    for (int s = 1; s <= log2(N); s++) {
        int m = 1 << s;
        float wm_real = cos(2 * M_PI / m);
        float wm_imag = -sin(2 * M_PI / m);

        for (int k = 0; k < N; k += m) {
            float w_real = 1.0;
            float w_imag = 0.0;
            for (int j = 0; j < m / 2; j++) {
                float t_real = w_real * real[k + j + m / 2] - w_imag * imag[k + j + m / 2];
                float t_imag = w_real * imag[k + j + m / 2] + w_imag * real[k + j + m / 2];
                float u_real = real[k + j];
                float u_imag = imag[k + j];

                real[k + j] = u_real + t_real;
                imag[k + j] = u_imag + t_imag;
                real[k + j + m / 2] = u_real - t_real;
                imag[k + j + m / 2] = u_imag - t_imag;

                float temp_real = w_real * wm_real - w_imag * wm_imag;
                w_imag = w_real * wm_imag + w_imag * wm_real;
                w_real = temp_real;
                //迭代W（n,k）
            }
        }
    }
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
    LARGE_INTEGER start;
    LARGE_INTEGER end;
    LARGE_INTEGER frequency;
    _Alignas(32) float real[FFT_SIZE];
    _Alignas(32) float imag[FFT_SIZE];
    _Alignas(32) float work_real[FFT_SIZE];
    _Alignas(32) float work_imag[FFT_SIZE];
    double total_us = 0.0;

    init_input(real, imag, FFT_SIZE);
    if (!QueryPerformanceFrequency(&frequency)) {
        fprintf(stderr, "QueryPerformanceFrequency failed\n");
        return 1;
    }

    /* 不计时预热。 */
    memcpy(work_real, real, sizeof(real));
    memcpy(work_imag, imag, sizeof(imag));
    fft_avx256(work_real, work_imag, FFT_SIZE);

    for (int r = 0; r < REPEAT; r++) {
        memcpy(work_real, real, sizeof(real));
        memcpy(work_imag, imag, sizeof(imag));

        QueryPerformanceCounter(&start);
        fft_avx256(work_real, work_imag, FFT_SIZE);
        QueryPerformanceCounter(&end);

        total_us +=
            (double)(end.QuadPart - start.QuadPart)
            * 1000000.0
            / (double)frequency.QuadPart;
    }

    printf("\nAVX2 FFT OUT (first %d bins)\n", OUTPUT_COUNT);
    for (int k = 0; k < OUTPUT_COUNT; k++) {
        printf("X[%d] = %f %+.6fi\n", k, work_real[k], work_imag[k]);
    }
    printf("\nN = %d, repeat = %d\n", FFT_SIZE, REPEAT);
    printf("AVX2 fft_avx256 average time: %.6f us\n",
           total_us / (double)REPEAT);

    return 0;
}
