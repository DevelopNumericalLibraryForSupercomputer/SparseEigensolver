#pragma once
#include "../../Device.hpp"
#include "../LinearOp.hpp"
#include "Utility.hpp"
#include "mkl.h"

namespace SE{
typedef MKL_INT MDESC[ 9 ];
const MKL_INT i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
MKL_INT info;
const double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;
/*
//memory managament
template<>
double* malloc<double, DEVICETYPE::MPI>(const int size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}

template<>
void free<double, DEVICETYPE::MPI>(double* ptr) {
    std::free(ptr);
}
*/

//ignore COPYTYPE: always NONE
template<>
void memcpy<int, DEVICETYPE::MPI>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<int, DEVICETYPE::MPI>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template<>
void memcpy<double, DEVICETYPE::MPI>(double* dest, const double* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(double));
}

template <>
void scal<double, DEVICETYPE::MPI>(const int n, const double alpha, double *x, const int incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void axpy<double, DEVICETYPE::MPI>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}

template <>
double nrm2<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx){
    return pddnrm2(n, x, incx);
	assert (incx==0);
    MDESC   descA;
    work = (double*) mkl_calloc(mp, sizeof( double ), 64);
    descinit( descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
	assert (info==0);
	pdlange('f', &n, &i_one, x, &i_zero, &i_zero, descA, work  ) //frobius norm
	mkl_free(work);
    return cblas_dnrm2(n, x, incx);
}

template <>
void gemv<double, DEVICETYPE::MPI>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    return cblas_dgemv(map_order_blas_MPI(order), map_transpose_blas_MPI(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template<>
void gemm<double, DEVICETYPE::MPI>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    return cblas_dgemm(map_order_blas_MPI(order), map_transpose_blas_MPI(transa), map_transpose_blas_MPI(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


//template<>
//int geqrf<double, DEVICETYPE::MPI>(const LAYOUT layout, int m, int n, double* a, int lda, double* tau){
//    return LAPACKE_dgeqrf(map_layout_lapack_MPI(layout), m, n, a, lda, tau);
//}
//
//template<>
//int orgqr<double, DEVICETYPE::MPI>(const LAYOUT layout, int m, int n, double* a, int lda, double* tau){
//    return LAPACKE_dorgqr(map_layout_lapack_MPI(layout), m, m, n, a, lda, tau);
//}
//
//template <>
//int geev<double, DEVICETYPE::MPI>(const LAYOUT layout, char jobvl, char jobvr, const int n, double* a, const int lda,
//          double* wr, double* wi, double* vl, const int ldvl, double* vr, const int ldvr){
//    return LAPACKE_dgeev(map_layout_lapack_MPI(layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
//}
//template <>
//int syev<double, DEVICETYPE::MPI>(const LAYOUT layout, char jobz, char uplo, const int n, double* a, const int lda, double* w){
//    return LAPACKE_dsyev(map_layout_lapack_MPI(layout), jobz, uplo, n, a, lda, w);
//}


}
