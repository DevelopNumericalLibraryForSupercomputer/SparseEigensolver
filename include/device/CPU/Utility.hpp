//ChatGPT3.5
#pragma once
#include "../../Utility.hpp"
#include <complex>
//#define MKL_Complex16 std::complex<double>
//#define MKL_Complex8  std::complex<float>
#include "mkl.h"

namespace SE{
template<>
double* malloc<double, CPU>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}
template<>
void free<double, CPU>(double* ptr) {
    std::free(ptr);
}
template<>
void memcpy<double, CPU>(double* dest, const double* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

/* templated version of mkl wrapper.
 * Containing functions
 * BLAS
 *  gemm
 *  daxpy
enum type variables
enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
 */

CBLAS_LAYOUT map_layout(SE_layout layout){
    switch (layout){
        case Blas_RowMajor: return CblasRowMajor;
        case Blas_ColMajor: return CblasColMajor;
    }
    exit(-1);
}
CBLAS_TRANSPOSE map_transpose(SE_transpose trans){
    switch (trans){
        case Blas_NoTrans:   return CblasNoTrans;
        case Blas_Trans:     return CblasTrans;
        case Blas_ConjTrans: return CblasConjTrans;
    }
    exit(-1);
}

template<>
void gemm<double, CPU>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    return cblas_dgemm(map_layout(Layout), map_transpose(transa), map_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
/*
template <>
void gemm<std::complex<double>, CPU>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                                     const size_t m, const size_t n, const size_t k,
                                     const std::complex<double> alpha, const std::complex<double> *a, const size_t lda,
                                     const std::complex<double> *b, const size_t ldb, const std::complex<double> beta,
                                     std::complex<double> *c, const size_t ldc){
    return cblas_zgemm(map_layout(Layout), map_transpose(transa), map_transpose(transb),
                       m, n, k, (const void *)&alpha, (const void *)a, lda, (const void *)b, ldb, (const void *)&beta, (void *)c, ldc);
}
*/
template <>
void axpy<double, CPU>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}


}