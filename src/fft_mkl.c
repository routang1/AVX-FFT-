#include "fft_mkl.h"

#include <stdio.h>

static int report_mkl_error(MKL_LONG status, const char *api_name)
{
    if (!DftiErrorClass(status, DFTI_NO_ERROR)) {
        fprintf(stderr, "%s failed: %s\n", api_name, DftiErrorMessage(status));
        return 0;
    }
    return 1;
}

int fft_mkl_init(FFT_MKL_PLAN *plan, int N)
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

    return 1;
}

int fft_mkl_forward(FFT_MKL_PLAN *plan, float *real, float *imag)
{
    MKL_LONG status = DftiComputeForward(plan->descriptor, real, imag);
    return report_mkl_error(status, "DftiComputeForward");
}

int fft_mkl_destroy(FFT_MKL_PLAN *plan)
{
    MKL_LONG status = DftiFreeDescriptor(&plan->descriptor);
    return report_mkl_error(status, "DftiFreeDescriptor");
}
