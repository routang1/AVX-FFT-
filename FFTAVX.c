#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#define REPEAT 10000
#endif

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
    float nn = N;
    int* pn = (int*)&nn;
    int log2N = ((*pn)>>23)-127;

    for (int s = 1; s <= log2N; s++) {
        int m = 1 << s;
        float w_real_vec[8],w_imag_vec[8];
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
            float wm_real = cos(2 * M_PI / m);
            float wm_imag = -sin(2 * M_PI / m);
            float w8_real,w8_imag;
            
            for (int k = 0; k < N; k += m) {
                w_real_vec[0] = 1.0f;
                w_imag_vec[0] = 0.0f;
                w_real_vec[1] = wm_real;
                w_imag_vec[1] = wm_imag;
                w_real_vec[2] = w_real_vec[1] * wm_real - w_imag_vec[1] * wm_imag;
                w_imag_vec[2] = w_real_vec[1] * wm_imag +  w_imag_vec[1] * wm_real;
                w_real_vec[3] = w_real_vec[2] * wm_real - w_imag_vec[2] * wm_imag;
                w_imag_vec[3] = w_real_vec[2] * wm_imag +  w_imag_vec[2] * wm_real;
                w_real_vec[4] = w_real_vec[3] * wm_real - w_imag_vec[3] * wm_imag;
                w_imag_vec[4] = w_real_vec[3] * wm_imag +  w_imag_vec[3] * wm_real;
                w_real_vec[5] = w_real_vec[4] * wm_real - w_imag_vec[4] * wm_imag;
                w_imag_vec[5] = w_real_vec[4] * wm_imag +  w_imag_vec[4] * wm_real;
                w_real_vec[6] = w_real_vec[5] * wm_real - w_imag_vec[5] * wm_imag;
                w_imag_vec[6] = w_real_vec[5] * wm_imag +  w_imag_vec[5] * wm_real;
                w_real_vec[7] = w_real_vec[6] * wm_real - w_imag_vec[6] * wm_imag;
                w_imag_vec[7] = w_real_vec[6] * wm_imag +  w_imag_vec[6] * wm_real;

                w8_real = w_real_vec[7] * wm_real - w_imag_vec[7] * wm_imag;
                w8_imag = w_real_vec[7] * wm_imag +  w_imag_vec[7] * wm_real;
                __m256 w8_real_vec = _mm256_set1_ps(w8_real);
                __m256 w8_imag_vec = _mm256_set1_ps(w8_imag);

                for (int j = 0; j < m / 2; j+=8) {
                    __m256 w_real = _mm256_load_ps(w_real_vec);
                    __m256 w_imag = _mm256_load_ps(w_imag_vec);
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
                    w_real =_mm256_mul_ps(w_real,w8_real_vec);
                    w_imag =_mm256_mul_ps(w_imag,w8_imag_vec); 
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






int main(void)
{
    LARGE_INTEGER start, end, freq;
    int N = 8;
    int repeat = 1000;
    _Alignas(32) float real[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    _Alignas(32) float imag[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    _Alignas(32) float work_real[8];
    _Alignas(32) float work_imag[8];

    double total_us = 0.0;

    for (int r = 0; r < repeat; r++)
    {
        memcpy(work_real, real, sizeof(real));
        memcpy(work_imag, imag, sizeof(imag));
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&start);
        //fft(work_real, work_imag, N);
        fft_avx256(work_real, work_imag, N);

        QueryPerformanceCounter(&end);

        total_us +=
            (double)(end.QuadPart - start.QuadPart)
            * 1000000.0
            / (double)freq.QuadPart;
    }

    double avg_us = total_us ;
    printf("freq = %lld\n", freq.QuadPart);
    printf("delta = %lld\n", end.QuadPart - start.QuadPart);
    printf("\nFFT OUT\n");
    for (int k = 0; k < N; k++) {
        printf("X[%d] = %f %+.6fi\n",k, work_real[k], work_imag[k]);
    }
    printf("FFT运行时间：%.6f us\n",avg_us);

    

    return 0;
}