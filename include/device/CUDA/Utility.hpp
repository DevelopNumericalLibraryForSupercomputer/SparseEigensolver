#pragma once
#include "../../Utility.hpp"
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


cublasOperation_t map_transpose(SE::SE_transpose trans){
    switch (trans){
        case SE::NoTrans:   return CUBLAS_OP_N;
        case SE::Trans:     return CUBLAS_OP_T;
        case SE::ConjTrans: return CUBLAS_OP_C;
        default: throw std::runtime_error("map_transpose in device/CUDA/Utility.hpp");
    }

    // to supress warning;
    return CUBLAS_OP_N;
}
namespace SE
{

cublasHandle_t cublasHandle;

template<>
double* malloc<double, computEnv::CUDA>(const size_t size){
    double* ptr;
    cudaMalloc( &ptr, size*sizeof(double) );
    return static_cast<double*> (ptr);
}
template<>
void free<double, computEnv::CUDA>(double* ptr ){
    cudaFree(ptr);
    return;
}
template<>
void memcpy<double, computEnv::CUDA>(double* dest, const double* source, size_t size ){
    cudaMemcpyKind kind = cudaMemcpyDeviceToDevice ;
    cudaMemcpy(dest, source, sizeof(double)*size, kind );
    return;
}
template<>
void memset<double, computEnv::CUDA>(double* dest, int value, size_t size){
    cudaMemset((void*)dest, value, size * sizeof(double) );
    return;
}

template<>
void gemm<double, computEnv::CUDA>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    cublasDgemm(cublasHandle,  map_transpose(transa), map_transpose(transb), m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc);
    return;
}

template <>
void axpy<double, computEnv::CUDA>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    cublasDaxpy(cublasHandle, n, &a, x, incx, y, incy);
    return;
}

template <>
int geev<double, computEnv::CUDA>(const SE_layout Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return 0;
//    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}


}

