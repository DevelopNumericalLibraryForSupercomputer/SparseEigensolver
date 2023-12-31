#pragma once
//#include <iostream>
//#include <cstdlib>
#include <cstring>
//#include <numeric>
//#include <iterator>
//#include <algorithm>
//#include <array>

#include "../Type.hpp"

namespace SE{

//memory managament
template<typename DATATYPE, DEVICETYPE devicetype =DEVICETYPE::BASE>
DATATYPE* malloc(const size_t size){
    return static_cast<DATATYPE*>(std::malloc(size * sizeof(DATATYPE)));
}

template< DEVICETYPE devicetype=DEVICETYPE::BASE>
void free(void* ptr){
    std::free(ptr);
}

template<typename DATATYPE, DEVICETYPE devicetype=DEVICETYPE::BASE>
void memcpy(DATATYPE* dest, const DATATYPE* source, size_t size, COPYTYPE copy_type=COPYTYPE::NONE){
    std::cout << typeid(copy_type).name() <<"\t"<< typeid(COPYTYPE::NONE).name() <<std::endl;
    std::cout << (int) copy_type <<"\t" << (int) COPYTYPE::NONE <<std::endl;
    assert(COPYTYPE::NONE==(COPYTYPE) copy_type );
    std::memcpy(dest, source, size * sizeof(DATATYPE));
}

template<typename DATATYPE, DEVICETYPE devicetype=DEVICETYPE::BASE>
void memset(DATATYPE* dest, int value, size_t size){
    std::memset(dest, value, size * sizeof(DATATYPE));
}


//x = a * x
template <typename DATATYPE, DEVICETYPE devicetype>
void scal(const size_t n, const DATATYPE alpha, DATATYPE *x, const size_t incx);

//a * x + y
template <typename DATATYPE, DEVICETYPE devicetype>
void axpy(const size_t n, const DATATYPE a, const DATATYPE *x, const size_t incx, DATATYPE *y, const size_t incy);

//Euclidean norm, ||x||
template <typename DATATYPE, DEVICETYPE devicetype>
DATATYPE nrm2(const size_t n, const DATATYPE *x, const size_t incx);

//y (i*N+incy) = x (i*M+incx) 
template <typename DATATYPE, DEVICETYPE devicetype>
void copy(const size_t n, const DATATYPE *x, const size_t incx, DATATYPE *y, const size_t incy);

//alpha * A * x + beta * y
template <typename DATATYPE, DEVICETYPE devicetype>
void gemv(const ORDERTYPE layout, const TRANSTYPE transa, const size_t m, const size_t n, const DATATYPE alpha,
          const DATATYPE *a, const size_t lda, const DATATYPE *x, const size_t incx,
          const DATATYPE beta, DATATYPE *y, const size_t incy);

//alpha * A * x + b * y
//void coomv;
//void sparse mv
//see mkl_spblas.h  / for cuda, see cusparse
//todo - implement 
//void sparse_status_t mkl_sparse_d_mv ()

//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
//output : C
template <typename DATATYPE, DEVICETYPE devicetype>
void gemm(const ORDERTYPE layout, const TRANSTYPE transa, const TRANSTYPE transb, const size_t m, const size_t n, const size_t k,
          const DATATYPE alpha, const DATATYPE *a, const size_t lda,
          const DATATYPE *b, const size_t ldb, const DATATYPE beta,
          DATATYPE *c, const size_t ldc);

//Blas-like extensions
//void mkl_domatcopy (char ordering, char trans, size_t rows, size_t cols,
//                   const double alpha, const double * A, size_t lda, double * B, size_t ldb);
template <typename DATATYPE, DEVICETYPE devicetype>
void omatcopy(const ORDERTYPE layout, const TRANSTYPE trans, size_t rows, size_t cols,
              const double alpha, DATATYPE *a, size_t lda, DATATYPE *b, size_t ldb);


//LAPACK

//QR decomposition of general (possible negative diagonal) m by n matrix, without pivoting
//for the substentiall tall matrix, see geqr
//If we don't need to directly apply q matrix and just need to multiply with othter matrix, see ormqr
template <typename DATATYPE, DEVICETYPE devicetype>
int geqrf(const ORDERTYPE layout, size_t m, size_t n, DATATYPE* a, size_t lda, DATATYPE* tau);

//generate real orthogonal matrix Q from geqrf
//LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
//lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
//                           lapack_int k, double* a, lapack_int lda,
//                           const double* tau );
template <typename DATATYPE, DEVICETYPE devicetype>
int orgqr(const ORDERTYPE layout, size_t m, size_t n, DATATYPE* a, size_t lda, DATATYPE* tau);

//ormqr

//diagonalization of general n by n matrix
//lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          //lapack_int n, double* a, lapack_int lda, double* wr,
                          //double* wi, double* vl, lapack_int ldvl, double* vr,
//                          lapack_int ldvr );
template <typename DATATYPE, DEVICETYPE devicetype>
int geev(const ORDERTYPE layout, char jobvl, char jobvr, const size_t n, DATATYPE* a, const size_t lda,
          DATATYPE* wr, DATATYPE* wi, DATATYPE* vl, const size_t ldvl, DATATYPE* vr, const size_t ldvr);


template <typename DATATYPE, DEVICETYPE devicetype>
int syev(const ORDERTYPE layout, char jobz, char uplo, const size_t n, double* a, const size_t lda, double* w);
}
