#pragma once
#include "../../Device.hpp"
#include "../LinearOp.hpp"
#include "Utility.hpp"
#include "mkl.h"

namespace SE{
//memory managament
/*
template<>
double* malloc<double, DEVICETYPE::MKL>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));

template<>
void free<double, DEVICETYPE::MKL>(double* ptr) {
    std::free(ptr);
}
*/

//ignore COPYTYPE: always NONE
template<>
void memcpy<int, DEVICETYPE::MKL>(int* dest, const int* source, size_t size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<size_t, DEVICETYPE::MKL>(size_t* dest, const size_t* source, size_t size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(size_t));
}

template<>
void memcpy<double, DEVICETYPE::MKL>(double* dest, const double* source, size_t size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(double));
}

template <>
void scal<double, DEVICETYPE::MKL>(const size_t n, const double alpha, double *x, const size_t incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void axpy<double, DEVICETYPE::MKL>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}

template <>
void gemv<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const size_t m, const size_t n, const double alpha,
                       const double *a, const size_t lda, const double *x, const size_t incx,
                       const double beta, double *y, const size_t incy)
{
    return cblas_dgemv(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template<>
void gemm<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    return cblas_dgemm(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), map_transpose_blas_MKL(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template<>
int geqrf<double, DEVICETYPE::MKL>(const ORDERTYPE order, size_t m, size_t n, double* a, size_t lda, double* tau){
    return LAPACKE_dgeqrf(map_order_lapack_MKL(order), m, n, a, lda, tau);
}

template<>
int orgqr<double, DEVICETYPE::MKL>(const ORDERTYPE order, size_t m, size_t n, double* a, size_t lda, double* tau){
    return LAPACKE_dorgqr(map_order_lapack_MKL(order), m, m, n, a, lda, tau);
}

template <>
int geev<double, DEVICETYPE::MKL>(const ORDERTYPE order, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return LAPACKE_dgeev(map_order_lapack_MKL(order), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

template <>
int syev<double, DEVICETYPE::MKL>(const ORDERTYPE order, char jobz, char uplo, const size_t n, double* a, const size_t lda, double* w){
    return LAPACKE_dsyev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}



}
