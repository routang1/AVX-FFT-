#ifndef FFT_MKL_H
#define FFT_MKL_H

#include "mkl_dfti.h"

typedef struct {
    DFTI_DESCRIPTOR_HANDLE descriptor;
} FFT_MKL_PLAN;

int fft_mkl_init(FFT_MKL_PLAN *plan, int N);
int fft_mkl_forward(FFT_MKL_PLAN *plan, float *real, float *imag);
int fft_mkl_destroy(FFT_MKL_PLAN *plan);

#endif
