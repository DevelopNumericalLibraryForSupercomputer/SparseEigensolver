#pragma once
#include <complex>
#define MKL_Complex16 std::complex<double>
#include "../../Device.hpp"
#include "../LinearOp.hpp"
#include "Utility.hpp"
#include "mkl.h"

namespace SE{
//memory managament
template<>
double* malloc<double, DEVICETYPE::MKL>(const int size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}
template<>
void free <DEVICETYPE::MKL>(void* ptr){
    std::free(ptr);
}

//ignore COPYTYPE: always NONE

template<>
void memcpy<int, DEVICETYPE::MKL>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<double, DEVICETYPE::MKL>(double* dest, const double* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(double));
}

template<>
void memcpy<std::complex<double>, DEVICETYPE::MKL>(std::complex<double>* dest, const std::complex<double>* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(std::complex<double>));
}

template <>
void scal<double, double, DEVICETYPE::MKL>(const int n, const double alpha, double *x, const int incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void scal<double, std::complex<double>, DEVICETYPE::MKL>(const int n, const double alpha, std::complex<double> *x, const int incx){
    return cblas_zdscal(n, alpha, x, incx);
}


template <>
void scal<std::complex<double>, std::complex<double>, DEVICETYPE::MKL>(const int n, const std::complex<double> alpha, std::complex<double> *x, const int incx){
    return cblas_zscal(n, &alpha, x, incx);
}


template <>
void axpy<double, DEVICETYPE::MKL>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}
template <>
void axpy<std::complex<double>, DEVICETYPE::MKL>(const int n, const std::complex<double> a, const std::complex<double> *x, const int incx, std::complex<double> *y, const int incy){
    return cblas_zaxpy(n, &a, x, incx, y, incy);
}
//y =alpha*Ax +beta*y
template <>
void sbmv<double, DEVICETYPE::MKL>(const ORDERTYPE order,
            const char uplo, const int n, const int k,
            const double alpha,
            const double *a, const int lda,
            const double *x, const int incx,
            const double beta,
            double *y, const int incy
        ){
    return cblas_dsbmv(map_order_blas_MKL(order), map_uplo_blas_MKL(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
double nrm2<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx){
    return cblas_dnrm2(n, x, incx);
}
template <>
double nrm2<std::complex<double>, DEVICETYPE::MKL>(const int n, const std::complex<double> *x, const int incx){
    return cblas_dznrm2(n, x, incx);
}

template <>
void copy<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx, double* y, const int incy){
    cblas_dcopy(n, x, incx, y, incy);
}

template <>
void copy<std::complex<double>, DEVICETYPE::MKL>(const int n, const std::complex<double> *x, const int incx, std::complex<double>* y, const int incy){
    cblas_zcopy(n, x, incx, y, incy);
}
template <>
void gemv<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    return cblas_dgemv(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template <>
void gemv<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const std::complex<double> alpha,
                       const std::complex<double> *a, const int lda, const std::complex<double> *x, const int incx,
                       const std::complex<double> beta, std::complex<double> *y, const int incy)
{
    return cblas_zgemv(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
}

template<>
void gemm<double, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    return cblas_dgemm(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), map_transpose_blas_MKL(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}
template<>
void gemm<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const std::complex<double> alpha, const std::complex<double> *a, const int lda,
                       const std::complex<double> *b, const int ldb, const std::complex<double> beta,
                       std::complex<double> *c, const int ldc){
    return cblas_zgemm(map_order_blas_MKL(order), map_transpose_blas_MKL(transa), map_transpose_blas_MKL(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
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
int geqrf<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, std::complex<double>* a, int lda, std::complex<double>* tau){
    return LAPACKE_zgeqrf(map_order_lapack_MKL(order), m, n, a, lda, tau);
}

template<>
int orgqr<double, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, double* a, int lda, double* tau){
    return LAPACKE_dorgqr(map_order_lapack_MKL(order), m, n, n, a, lda, tau);
}

template<>
int orgqr<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, std::complex<double>* a, int lda, std::complex<double>* tau){
    return LAPACKE_zungqr(map_order_lapack_MKL(order), m, n, n, a, lda, tau);
}


template <>
int geev<double, DEVICETYPE::MKL>(const ORDERTYPE order, const char jobvl, const char jobvr, const int n, double* a, const int lda,
          std::complex<double>* w, double* vl, const int ldvl, double* vr, const int ldvr){

    double* wr = new double[n];
    double* wi = new double[n];
    auto return_val =  LAPACKE_dgeev(map_order_lapack_MKL(order), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
    #pragma omp parallel for
    for (int i=0; i<n; i++){
        w[i] = std::complex<double>(wr[i],wi[i]);
    }
    return return_val;
}

template <>
int geev<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, const char jobvl, const char jobvr, const int n, std::complex<double>* a, const int lda,
          std::complex<double>* w, std::complex<double>* vl, const int ldvl, std::complex<double>* vr, const int ldvr){
    // using only wi
    // wr should not be initialized
    return LAPACKE_zgeev(map_order_lapack_MKL(order), jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}


template <>
int syev<double, DEVICETYPE::MKL>(const ORDERTYPE order, const char jobz, const char uplo, const int n, double* a, const int lda, double* w){
    return LAPACKE_dsyev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}

template <>
int syev<std::complex<double>, DEVICETYPE::MKL>(const ORDERTYPE order, const char jobz, const char uplo, const int n, std::complex<double>* a, const int lda, double* w){
    return LAPACKE_zheev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}


template <>
void vMul<double, DEVICETYPE::MKL>(const int n, const double* a, const double* b, double* y){
    return vdMul(n,a,b,y);  
}

template <>
void vMul<std::complex<double>, DEVICETYPE::MKL>(const int n, const std::complex<double>* a, const std::complex<double>* b, std::complex<double>* y){
    return vzMul(n,a,b,y);  
}


}
