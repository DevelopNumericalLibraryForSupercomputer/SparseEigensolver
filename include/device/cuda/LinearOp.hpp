#pragma once
#include "../LinearOp.hpp"
#include "Utility.hpp"

namespace SE{

namespace SE
{

cublasHandle_t handle=NULL;
cudaStream_t stream=NULL;

template<>
double* malloc<double, DEVICETYPE::CUDA>(const int size){
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
void memcpy<double, DEVICETYPE::CUDA>(double* dest, const double* source, int size, COPYTYPE copy_type ){
    //cudaMemcpyKind kind = cudaMemcpyDeviceToDevice ;
    cudaMemcpy(dest, source, sizeof(double)*size, map_copytype(copy_type) );
    return;
}
template<>
void memset<double, DEVICETYPE::CUDA>(double* dest, int value, int size){
    cudaMemset((void*)dest, value, size * sizeof(double) );
    return;
}

template <>
void scal<double, DEVICETYPE::CUDA>(const int n, const double alpha, double *x, const int incx){
    cublasDscal(handle, n, &alpha, x, incx);
    return;
}

template <>
void axpy<double, DEVICETYPE::CUDA>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    cublasDaxpy(handle, n, &a, x, incx, y, incy);
    return;
}

template <>
double nrm2<double, DEVICETYPE::CUDA>(const int n, const double *x, const int incx){
    double result = 0.0;
    cublasDnrm2(handle, n, x, incx, &result);
    return result;
}

template <>
void copy<double, DEVICETYPE::CUDA>(const int n, const double *x, const int incx, double* y, const int incy){
    cublasDcopy(handle, n, x, incx, y, incy);
    return;
}

template <>
void gemv<double, DEVICETYPE::CUDA>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    cublasDgemv(handle, map_transtype(transa), m, n, &alpha, a, lda, x, incx, &beta, y, incy);
    return;
}

template<>
void gemm<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    cublasDgemm(handle,  map_transtype(transa), map_transtype(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    return;
}

template<>
int geqrf<double, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, double* a, int lda, double* tau){
    return 0;
}

template<>
int orgqr<double, DEVICETYPE::MKL>(const ORDERTYPE order, int m, int n, double* a, int lda, double* tau){
    return 0;
}

template <>
int geev<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, char jobvl, char jobvr, const int n, double* a, const int lda,
          double* wr, double* wi, double* vl, const int ldvl, double* vr, const int ldvr){
    return 0;
//    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

template <>
int syev<double, DEVICETYPE::CUDA>(const ORDERTYPE order, char jobz, char uplo, const int n, double* a, const int lda, double* w){
    return 0;
    //return LAPACKE_dsyev(map_order_lapack_MKL(order), jobz, uplo, n, a, lda, w);
}


}

