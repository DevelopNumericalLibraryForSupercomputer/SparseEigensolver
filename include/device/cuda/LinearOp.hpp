#pragma once
#include "../LinearOp.hpp"
#include "Utility.hpp"

namespace SE{

namespace SE
{

cublasHandle_t handle=NULL;
cudaStream_t stream=NULL;

template<>
double* malloc<double, DEVICETYPE::CUDA>(const size_t size){
    double* ptr;
    cudaMalloc( &ptr, size*sizeof(double) );
    return static_cast<double*> (ptr);
}
template<>
void free<DEVICETYPE::CUDA>(void* ptr ){
    cudaFree(ptr);
    return;
}
template<>
void memcpy<double, DEVICETYPE::CUDA>(double* dest, const double* source, size_t size, COPYTYPE copy_type ){
    //cudaMemcpyKind kind = cudaMemcpyDeviceToDevice ;
    cudaMemcpy(dest, source, sizeof(double)*size, map_copytype(copy_type) );
    return;
}
template<>
void memset<double, DEVICETYPE::CUDA>(double* dest, int value, size_t size){
    cudaMemset((void*)dest, value, size * sizeof(double) );
    return;
}

template <>
void scal<double, DEVICETYPE::CUDA>(const size_t n, const double alpha, double *x, const size_t incx){
    cublasDscal(handle, n, &alpha, x, incx);
    return;
}

template <>
void axpy<double, DEVICETYPE::CUDA>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    cublasDaxpy(handle, n, &a, x, incx, y, incy);
    return;
}

template <>
double nrm2<double, DEVICETYPE::CUDA>(const size_t n, const double *x, const size_t incx){
    double result = 0.0;
    cublasDnrm2(handle, n, x, incx, &result);
    return result;
}

template <>
void copy<double, DEVICETYPE::CUDA>(const size_t n, const double *x, const size_t incx, double* y, const size_t incy){
    cublasDcopy(handle, n, x, incx, y, incy);
    return;
}

template <>
void gemv<double, DEVICETYPE::CUDA>(const ORDERTYPE order, const TRANSTYPE transa, const size_t m, const size_t n, const double alpha,
                       const double *a, const size_t lda, const double *x, const size_t incx,
                       const double beta, double *y, const size_t incy)
{
    cublasDgemv(handle, map_transtype(transa), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    return;
}

template<>
void gemm<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, const TRANSTYPE transa, const TRANSTYPE transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    cublasDgemm(handle,  map_transtype(transa), map_transtype(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    return;
}

template<>
int geqrf<double, DEVICETYPE::MKL>(const ORDERTYPE order, size_t m, size_t n, double* a, size_t lda, double* tau){
    return 0;
}

template<>
int orgqr<double, DEVICETYPE::MKL>(const ORDERTYPE order, size_t m, size_t n, double* a, size_t lda, double* tau){
    return 0;
}

template <>
int geev<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return 0;
//    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

template <>
int syev<double, DEVICETYPE::CUDA>(const ORDERTYPE order, char jobz, char uplo, const size_t n, double* a, const size_t lda, double* w){
    return 0;
    //return LAPACKE_dsyev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}


}

