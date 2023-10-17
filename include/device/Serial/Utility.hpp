#pragma once
#include "../../Device.hpp"
#include "../../Utility.hpp"
#include "mkl.h"

namespace SE{
//memory managament
template<>
double* malloc<double, computEnv::MKL>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}

template<>
void free<double, computEnv::MKL>(double* ptr) {
    std::free(ptr);
}

template<>
void memcpy<double, computEnv::MKL>(double* dest, const double* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

CBLAS_LAYOUT map_layout_blas(SE_layout layout){
    switch (layout){
        case RowMajor: return CblasRowMajor;
        case ColMajor: return CblasColMajor;
    }
    exit(-1);
}

int map_layout_lapack(SE_layout layout){
    switch (layout){
        case RowMajor: return LAPACK_ROW_MAJOR;
        case ColMajor: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose(SE_transpose trans){
    switch (trans){
        case NoTrans:   return CblasNoTrans;
        case Trans:     return CblasTrans;
        case ConjTrans: return CblasConjTrans;
    }
    exit(-1);
}

template<>
void gemm<double, computEnv::MKL>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    return cblas_dgemm(map_layout_blas(Layout), map_transpose(transa), map_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
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
void axpy<double, computEnv::MKL>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}

template <>
int geev<double, computEnv::MKL>(const SE_layout Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}


}