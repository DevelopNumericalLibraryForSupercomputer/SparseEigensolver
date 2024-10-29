#pragma once 
#include <cassert>
#include <complex>
#define MKL_Complex16 std::complex<double>
#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "device/LinearOp.hpp"
#include "Device.hpp"
//#include "mkl.h"
#include "UtilDevice.hpp"
#include "Utility.hpp"

namespace SE{

////////////////////// SYCL device control
std::unique_ptr<sycl::queue> p_que;
std::vector<cl::sycl::event> events;

void set_device(const sycl::device device){
    p_que = std::make_unique<sycl::queue>(device);    
}


template<>
double* malloc<double, DEVICETYPE::SYCL>(const int size){
    return cl::sycl::malloc_device<double>(size, *p_que);
}
template<>
std::complex<double>* malloc<std::complex<double>, DEVICETYPE::SYCL>(const int size){
    return cl::sycl::malloc_device<std::complex<double> >  (size, *p_que); 
}
template<>
void free<DEVICETYPE::SYCL>(void* ptr){
    cl::sycl::free(ptr,*p_que);
    return;
}
template<>
void memcpy<double,DEVICETYPE::SYCL>(double* dest, const double* source, int size, COPYTYPE copy_type){
    auto event = p_que->memcpy(dest, source, size * sizeof(double));
    events.push_back(event);
    return;
}
template<>
void memcpy<std::complex<double>,DEVICETYPE::SYCL>(std::complex<double>* dest, const std::complex<double>* source, int size, COPYTYPE copy_type){
    auto event = p_que->memcpy(dest, source, size * sizeof(std::complex<double>));
    events.push_back(event);
    return;
}
template<>
void memset<double, DEVICETYPE::SYCL>(double* dest, const int value, const int size){
    auto event = p_que->memset(dest, value, size * sizeof(double));
    events.push_back(event);
    return;
}

template<>
void scal<double, double, DEVICETYPE::SYCL>(const int n, const double alpha, double *x, const int incx){
    auto event = oneapi::mkl::blas::row_major::scal(*p_que, n,alpha, x, incx, events);
    events.push_back(event);
    return;
}

template<>
void axpy<double, DEVICETYPE::SYCL>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    auto event = oneapi::mkl::blas::row_major::axpy(*p_que, n,a, x, incx, y, incy, events);
    events.push_back(event);
    return;
}

template <>
void sbmv<double, DEVICETYPE::SYCL>(
            const ORDERTYPE layout,
            const char uplo, const int n, const int k,
            const double alpha,
            const double *a, const int lda,
            const double *x, const int incx,
            const double beta,
            double *y, const int incy
        ){
    if (layout==ORDERTYPE::ROW){
        auto event = oneapi::mkl::blas::row_major::sbmv(*p_que, map_uplo_SYCL(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy, events);
        events.push_back(event);
    }
    else{
        auto event = oneapi::mkl::blas::column_major::sbmv(*p_que, map_uplo_SYCL(uplo), n, k, alpha, a, lda, x, incx, beta, y, incy, events);
        events.push_back(event);
    }
    return;
}

template <>
double nrm2<double, DEVICETYPE::SYCL>(const int n, const double *x, const int incx){
    double result;
    auto event = oneapi::mkl::blas::row_major::nrm2(*p_que, n, x, incx, &result, events);
    events.push_back(event);
    return result;
}
template <>
void copy<double, DEVICETYPE::SYCL>(const int n, const double *x, const int incx, double* y, const int incy){
    auto event = oneapi::mkl::blas::row_major::copy(*p_que, n, x, incx, y, incy, events);
    events.push_back(event);
    return;
}

template<>
void gemv<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    if (layout==ORDERTYPE::ROW){
        auto event = oneapi::mkl::blas::row_major::gemv(*p_que, map_transpose_SYCL(transa), m, n, alpha, a, lda, x, incx, beta, y, incy, events );
        events.push_back(event);
    }
    else{
        auto event = oneapi::mkl::blas::column_major::gemv(*p_que, map_transpose_SYCL(transa), m, n, alpha, a, lda, x, incx, beta, y, incy, events );
        events.push_back(event);
        
    }
    return;
}
template<>
void gemm<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    if (layout==ORDERTYPE::ROW){
        auto event = oneapi::mkl::blas::row_major::gemm(*p_que, map_transpose_SYCL(transa), map_transpose_SYCL(transb), m,n,k, alpha, a, lda, b, ldb, beta, c, ldc, events);
        events.push_back(event);
    }
    else{
        auto event = oneapi::mkl::blas::column_major::gemm(*p_que, map_transpose_SYCL(transa), map_transpose_SYCL(transb), m,n,k, alpha, a, lda, b, ldb, beta, c, ldc, events);
        events.push_back(event);
    }
    return;
}

template <>
int geqrf<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, int m, int n, double* a, int lda, double* tau){
    assert (layout==ORDERTYPE::COL);
    const auto scratchpad_size = oneapi::mkl::lapack::geqrf_scratchpad_size<double>(*p_que, m,n,lda);
    auto scratchpad      = malloc<double, DEVICETYPE::SYCL>(scratchpad_size);
    auto event=oneapi::mkl::lapack::geqrf(*p_que, m, n, a, lda, tau, scratchpad, scratchpad_size, events);    
    events.push_back(event);
    return 0;
}

template <>
int orgqr<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, int m, int n, double* a, int lda, double* tau){
    assert (layout==ORDERTYPE::COL);
    const auto scratchpad_size = oneapi::mkl::lapack::orgqr_scratchpad_size<double>(*p_que, m,n,n, lda);
    auto scratchpad      = malloc<double, DEVICETYPE::SYCL>(scratchpad_size);
    auto event=oneapi::mkl::lapack::orgqr(*p_que, m, n, n, a, lda, tau, scratchpad, scratchpad_size, events);    
    events.push_back(event);
    return 0;
}

//template<>
//int geev<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, const char jobvl, const char jobvr, const int n, DATATYPE* a, const int lda,
//          std::complex<typename real_type<DATATYPE>::type>* w, DATATYPE* vl, const int ldvl, DATATYPE* vr, const int ldvr){
//    
//    
//}
template <>
int syev<double, DEVICETYPE::SYCL>(const ORDERTYPE layout, const char jobz, const char uplo, const int n, double* a, const int lda, double* w){

    assert (layout==ORDERTYPE::COL);
    auto jobz_ = map_jobz_SYCL(jobz);
    auto uplo_ = map_uplo_SYCL(uplo);

    const auto scratchpad_size = oneapi::mkl::lapack::syevd_scratchpad_size<double>( *p_que, jobz_, uplo_, n, lda);
    auto scratchpad      = malloc<double, DEVICETYPE::SYCL>(scratchpad_size);
    auto event           = oneapi::mkl::lapack::syevd(*p_que, jobz_, uplo_, n, a, lda, w, scratchpad, scratchpad_size, events);
    events.push_back(event);
    return 0;
}
template <>
void vMul<double, DEVICETYPE::SYCL>(const int n, const double* a, const double* b, double* y){
    auto event =  oneapi::mkl::vm::mul(*p_que,n,a,b,y, events);  
    events.push_back(event);
    return ;
}




//a * x + y








}
