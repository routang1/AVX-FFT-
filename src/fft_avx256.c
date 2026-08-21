#include "fft_avx256.h"

#include <immintrin.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
                __m256 x_real = _mm256_load_ps(real + k);
                //从 real[k] 开始，连续读取 8 个 float，装进一个 256 位 AVX 寄存器 x_real。
                __m256 x_imag = _mm256_load_ps(imag + k);
                __m256 swapped_real = _mm256_permute_ps(x_real,0b10110001);
                __m256 swapped_imag = _mm256_permute_ps(x_imag,0b10110001);
                __m256 diff_sum_real = _mm256_addsub_ps(x_real,swapped_real);
                __m256 diff_sum_imag = _mm256_addsub_ps(x_imag,swapped_imag);
                __m256 rst_real = _mm256_permute_ps(diff_sum_real,0b10110001);
                __m256 rst_imag = _mm256_permute_ps(diff_sum_imag,0b10110001);
                _mm256_store_ps(real + k,rst_real);
                _mm256_store_ps(imag + k,rst_imag);
            }
        }
        else if(m==4)
        {
            __m256 real_sign = _mm256_setr_ps(
                0.0f, 0.0f, -0.0f, -0.0f,
                0.0f, 0.0f, -0.0f, -0.0f
            );
            __m256 imag_sign = _mm256_setr_ps(
                0.0f, -0.0f, -0.0f, 0.0f,
                0.0f, -0.0f, -0.0f, 0.0f
            );
            for (int k = 0; k < N; k += 8) {
                __m256 x_real = _mm256_load_ps(real + k);
                __m256 x_imag = _mm256_load_ps(imag + k);
                __m256 a_real = _mm256_shuffle_ps(x_real,x_real,0b01000100);
                __m256 a_imag = _mm256_shuffle_ps(x_imag,x_imag,0b01000100);
                __m256 b_parts = _mm256_unpackhi_ps(x_real,x_imag);
                __m256 mixed_real = _mm256_shuffle_ps(b_parts,b_parts,0b11001100);
                __m256 mixed_imag = _mm256_shuffle_ps(b_parts,b_parts,0b10011001);
                __m256 r_real = _mm256_xor_ps(mixed_real,real_sign);
                __m256 r_imag = _mm256_xor_ps(mixed_imag,imag_sign);
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
                    __m256 r_real = _mm256_fmsub_ps(w_real,x2_real,_mm256_mul_ps(w_imag,x2_imag));
                    __m256 r_imag = _mm256_fmadd_ps(w_real,x2_imag,_mm256_mul_ps(w_imag,x2_real));
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
