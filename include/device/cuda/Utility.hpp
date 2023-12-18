#pragma once
#include "../LinearOp.hpp"
//#include "../../Utility.hpp"
#include <cublas_v2.h>
/*
cublas matrix multiplication에 대한 설명은 
https://github.com/zchee/cuda-sample/blob/master/0_Simple/matrixMulCUBLAS/matrixMulCUBLAS.cpp
https://docs.nvidia.com/cuda/cublas/index.html?highlight=gemm#cublas-level-3-function-reference

custum enum type variables
matrix transpose
typedef enum {
    CUBLAS_OP_N=0,  
    CUBLAS_OP_T=1,  
    CUBLAS_OP_C=2  
} cublasOperation_t;

*/


cublasOperation_t map_transtype(SE::TRANSTYPE trans){
    switch (trans){
        case SE::TRANSTYPE::N:     return CUBLAS_OP_N;
        case SE::TRANSTYPE::T:     return CUBLAS_OP_T;
        case SE::TRANSTYPE::C:     return CUBLAS_OP_C;
        default: throw std::runtime_error("map_transtype in device/CUDA/Utility.hpp");
    }

    // to supress warning;
    return CUBLAS_OP_N;
}

cudaMemcpyKind map_copytype(SE::COPYTYPE copy_type){
    switch (copy_type){
        case SE::COPYTYPE::NONE:          return cudaMemcpyHostToHost;
        case SE::COPYTYPE::HOST2DEVICE:   return cudaMemcpyHostToDevice;
        case SE::COPYTYPE::DEVICE2HOST:   return cudaMemcpyDeviceToHost;
        case SE::COPYTYPE::DEVICE2DEVICE: return cudaMemcpyDeviceToDevice;
        default: throw std::runtime_error("map_copytype in device/CUDA/Utility.hpp");
    }
    return  cudaMemcpyHostToHost;
}
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

template<>
void gemm<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, const TRANSTYPE transa, const TRANSTYPE transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    cublasDgemm(handle,  map_transtype(transa), map_transtype(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    return;
}

template <>
void axpy<double, DEVICETYPE::CUDA>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    cublasDaxpy(handle, n, &a, x, incx, y, incy);
    return;
}

template <>
int geev<double, DEVICETYPE::CUDA>(const ORDERTYPE Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return 0;
//    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}


}

