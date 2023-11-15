#pragma once
#include "../../Device.hpp"
#include "../LinearOp.hpp"
#include "mkl.h"
#include "../MKL/LinearOp.hpp"

namespace SE{
//memory managament

template<>
double* malloc<double, MPI>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}
template<>
size_t* malloc<size_t, MPI>(const size_t size) {
    return static_cast<size_t*>(std::malloc(size * sizeof(size_t)));
}
template<>
int* malloc<int, MPI>(const size_t size) {
    return static_cast<int*>(std::malloc(size * sizeof(int)));
}


template<>
void free<double, MPI>(double* ptr) {
    std::free(ptr);
}
template<>
void free<size_t, MPI>(size_t* ptr) {
    std::free(ptr);
}
template<>
void free<int, MPI>(int* ptr) {
    std::free(ptr);
}

template<>
void memcpy<double, MPI>(double* dest, const double* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

template<>
void memset<double, MPI>(double* dest, int value, size_t size){
    std::memset(dest, value, size * sizeof(double));
}
template<>
void memset<size_t, MPI>(size_t* dest, int value, size_t size){
    std::memset(dest, value, size * sizeof(size_t));
}

/*
template<>
CBLAS_LAYOUT map_layout_blas<MPI>(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return CblasRowMajor;
        case SE_layout::ColMajor: return CblasColMajor;
    }
    exit(-1);
}
template<>
int map_layout_lapack<MPI>(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return LAPACK_ROW_MAJOR;
        case SE_layout::ColMajor: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}
template<>
CBLAS_TRANSPOSE map_transpose<MPI>(SE_transpose trans){
    switch (trans){
        case SE_transpose::NoTrans:   return CblasNoTrans;
        case SE_transpose::Trans:     return CblasTrans;
        case SE_transpose::ConjTrans: return CblasConjTrans;
    }
    exit(-1);
}

*/

template <>
void scal<double, MPI>(const size_t n, const double alpha, double *x, const size_t incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void axpy<double, MPI>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}
/*
template <>
void gemv<double, MKL>(const SE_layout layout, const SE_transpose transa, const size_t m, const size_t n, const double alpha,
                       const double *a, const size_t lda, const double *x, const size_t incx,
                       const double beta, double *y, const size_t incy)
{
    return cblas_dgemv(map_layout_blas(layout), map_transpose(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}
*/
template<>
void gemm<double, MPI>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    //double* temp_c = malloc<double, MPI>(m*n);
    //cblas_dgemm(map_layout_blas(Layout), map_transpose(transa), map_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    //allreduce<double, MPI> (temp_c, m*n, c, SEop::SUM);
}
/*
template<>
int geqrf<double, MKL>(const SE_layout layout, size_t m, size_t n, double* a, size_t lda, double* tau){
    return LAPACKE_dgeqrf(map_layout_lapack(layout), m, n, a, lda, tau);
}

template<>
int orgqr<double, MKL>(const SE_layout layout, size_t m, size_t n, double* a, size_t lda, double* tau){
    return LAPACKE_dorgqr(map_layout_lapack(layout), m, m, n, a, lda, tau);
}
*/
template <>
int geev<double, MPI>(const SE_layout Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}


}
