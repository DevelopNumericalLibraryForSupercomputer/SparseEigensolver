#pragma once
#include "../../Device.hpp"
#include "../LinearOp.hpp"
#include "Utility.hpp"
#include "mkl.h"

namespace SE{
//memory managament
/*
template<>
double* malloc<double, DEVICETYPE::MKL>(const int size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));

template<>
void free<double, DEVICETYPE::MKL>(double* ptr) {
    std::free(ptr);
}
*/

//ignore COPYTYPE: always NONE
template<>
void memcpy<int, DEVICETYPE::MKL>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<int, DEVICETYPE::MKL>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<double, DEVICETYPE::MKL>(double* dest, const double* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(double));
}

template <>
void scal<double, DEVICETYPE::MKL>(const int n, const double alpha, double *x, const int incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void axpy<double, DEVICETYPE::MKL>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}

template <>
double nrm2<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx){
    return cblas_dnrm2(n, x, incx);
}

template <>
void copy<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx, double* y, const int incy){
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
void gemv<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    return cblas_dgemv(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template<>
void gemm<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    return cblas_dgemm(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), map_transpose_blas_MKL(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

//template <>
//void omatcopy<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE trans, int rows, int cols,
//                                       const double alpha, double *a, int lda, double *b, int ldb){
//    return mkl_domatcopy(map_order_blas_extension_MKL(order), map_transpose_blas_extension_MKL(trans), rows, cols, alpha, a, lda, b, ldb);
//}

template<>
int geqrf<double, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, double* a, int lda, double* tau){
    return LAPACKE_dgeqrf(map_order_lapack_MKL(order), m, n, a, lda, tau);
}

template<>
int orgqr<double, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, double* a, int lda, double* tau){
    return LAPACKE_dorgqr(map_order_lapack_MKL(order), m, n, n, a, lda, tau);
}

template <>
int geev<double, DEVICETYPE::MKL>(const ORDERTYPE order, char jobvl, char jobvr, const int n, double* a, const int lda,
          double* wr, double* wi, double* vl, const int ldvl, double* vr, const int ldvr){
    return LAPACKE_dgeev(map_order_lapack_MKL(order), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

template <>
int syev<double, DEVICETYPE::MKL>(const ORDERTYPE order, char jobz, char uplo, const int n, double* a, const int lda, double* w){
    return LAPACKE_dsyev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}



}
