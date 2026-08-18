#include <stdio.h>
#include <math.h>
#include <immintrin.h>
#include <string.h>

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


static void print_complex(
    const char *title,
    const float *real,
    const float *imag,
    int N)
{
    printf("\n%s\n", title);

    for (int i = 0; i < N; i++)
    {
        printf(
            "[%2d] %12.6f %+.6fi\n",
            i,
            real[i],
            imag[i]
        );
    }
}


/* =========================================================
 * 比较标量 FFT 与 AVX FFT
 * ========================================================= */
static void compare_result(
    const float *scalar_real,
    const float *scalar_imag,
    const float *avx_real,
    const float *avx_imag,
    int N)
{
    printf("\n误差比较：\n");

    float max_error = 0.0f;

    for (int i = 0; i < N; i++)
    {
        float error_real =
            fabsf(scalar_real[i] - avx_real[i]);

        float error_imag =
            fabsf(scalar_imag[i] - avx_imag[i]);

        float error =
            error_real > error_imag
            ? error_real
            : error_imag;

        if (error > max_error)
            max_error = error;

        printf(
            "[%2d] real error = %.8f   imag error = %.8f\n",
            i,
            error_real,
            error_imag
        );
    }

    printf("\n最大误差 = %.10f\n", max_error);

    if (max_error < 1e-4f)
        printf("结果：PASS\n");
    else
        printf("结果：FAIL\n");
}


/* =========================================================
 * 通用测试函数
 * ========================================================= */
static void run_test(
    const char *name,
    const float *input,
    int N)
{
    /*
     * 你的 AVX 代码使用 _mm256_load_ps，
     * 所以这里显式使用 32 字节对齐。
     *
     * 同时至少分配 16 个元素，
     * 避免 N=4 时 AVX 一次读取 8 个 float 越界。
     */
    _Alignas(32) float scalar_real[16] = {0};
    _Alignas(32) float scalar_imag[16] = {0};

    _Alignas(32) float avx_real[16] = {0};
    _Alignas(32) float avx_imag[16] = {0};

    for (int i = 0; i < N; i++)
    {
        scalar_real[i] = input[i];
        avx_real[i] = input[i];

        scalar_imag[i] = 0.0f;
        avx_imag[i] = 0.0f;
    }

    printf("\n");
    printf("=================================================\n");
    printf("%s\n", name);
    printf("=================================================\n");

    print_complex(
        "原始输入：",
        scalar_real,
        scalar_imag,
        N
    );


    /* 标量 FFT */
    fft(
        scalar_real,
        scalar_imag,
        N
    );


    /* AVX FFT */
    fft_avx256(
        avx_real,
        avx_imag,
        N
    );


    print_complex(
        "标量 FFT 结果：",
        scalar_real,
        scalar_imag,
        N
    );


    print_complex(
        "AVX FFT 结果：",
        avx_real,
        avx_imag,
        N
    );


    compare_result(
        scalar_real,
        scalar_imag,
        avx_real,
        avx_imag,
        N
    );
}


/* =========================================================
 * main
 * ========================================================= */
int main(void)
{
    /* -------------------------
     * 4 点 FFT
     * ------------------------- */
    float input4[4] =
    {
        1.0f,
        2.0f,
        3.0f,
        4.0f
    };


    /* -------------------------
     * 8 点 FFT
     * ------------------------- */
    float input8[8] =
    {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
        5.0f,
        6.0f,
        7.0f,
        8.0f
    };


    /* -------------------------
     * 16 点 FFT
     * ------------------------- */
    float input16[16] =
    {
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f
    };


    run_test(
        "4 点 FFT 测试",
        input4,
        4
    );

    run_test(
        "8 点 FFT 测试",
        input8,
        8
    );

    run_test(
        "16 点 FFT 测试",
        input16,
        16
    );


    return 0;
}
