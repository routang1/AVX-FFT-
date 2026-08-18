#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
//交换函数
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
    int N = 8;

    // 输入信号：1,2,3,4,5,6,7,8
    float real[8] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };

    // 输入是纯实数，因此虚部全部为 0
    float imag[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    //printf("FFT 输入：\n");
    for (int i = 0; i < N; i++) {
        printf("x[%d] = %f + %fi\n",
               i, real[i], imag[i]);
    }

    // 执行 FFT
    fft(real, imag, N);

    printf("\nFFT OUT\n");
    for (int k = 0; k < N; k++) {
        printf("X[%d] = %f %+.6fi\n",
               k, real[k], imag[k]);
    }

    return 0;
}