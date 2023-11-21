#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <array>

namespace SE{
//memory managament
template<typename datatype, typename computEnv>
datatype* malloc(const size_t size){
    return static_cast<datatype*>(std::malloc(size * sizeof(datatype)));
}

template<typename datatype, typename computeEnv>
void free(datatype* ptr){
    std::free(ptr);
}

template<typename datatype, typename computeEnv>
void memcpy(datatype* dest, const datatype* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

template<typename datatype, typename computeEnv>
void memset(datatype* dest, int value, size_t size){
    std::memset(dest, value, size * sizeof(double));
}

//mkl - BLAS
enum class SE_transpose{
    NoTrans,
    Trans,
    ConjTrans
};
enum class SE_layout{
    RowMajor,
    ColMajor
};

//x = a * x
template <typename datatype, typename computeEnv>
void scal(const size_t n, const datatype alpha, datatype *x, const size_t incx);

//a * x + y
template <typename datatype, typename computeEnv>
void axpy(const size_t n, const datatype a, const datatype *x, const size_t incx, datatype *y, const size_t incy);

//alpha * A * x + beta * y
template <typename datatype, typename computeEnv>
void gemv(const SE_layout layout, const SE_transpose transa, const size_t m, const size_t n, const datatype alpha,
          const datatype *a, const size_t lda, const datatype *x, const size_t incx,
          const datatype beta, datatype *y, const size_t incy);

//alpha * A * x + b * y
//void coomv;
//void sparse mv
//see mkl_spblas.h  / for cuda, see cusparse
//todo - implement 
//void sparse_status_t mkl_sparse_d_mv ()

//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
//output : C
template <typename datatype, typename computeEnv>
void gemm(const SE_layout layout, const SE_transpose transa, const SE_transpose transb, const size_t m, const size_t n, const size_t k,
          const datatype alpha, const datatype *a, const size_t lda,
          const datatype *b, const size_t ldb, const datatype beta,
          datatype *c, const size_t ldc);



//LAPACK

//QR decomposition of general (possible negative diagonal) m by n matrix, without pivoting
//for the substentiall tall matrix, see geqr
//If we don't need to directly apply q matrix and just need to multiply with othter matrix, see ormqr
template <typename datatype, typename computeEnv>
int geqrf(const SE_layout layout, size_t m, size_t n, datatype* a, size_t lda, datatype* tau);

//generate real orthogonal matrix Q from geqrf
//LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
//lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
//                           lapack_int k, double* a, lapack_int lda,
//                           const double* tau );
template <typename datatype, typename computeEnv>
int orgqr(const SE_layout layout, size_t m, size_t n, datatype* a, size_t lda, datatype* tau);

//ormqr

//diagonalization of general n by n matrix
//lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          //lapack_int n, double* a, lapack_int lda, double* wr,
                          //double* wi, double* vl, lapack_int ldvl, double* vr,
//                          lapack_int ldvr );
template <typename datatype, typename computeEnv>
int geev(const SE_layout layout, char jobvl, char jobvr, const size_t n, datatype* a, const size_t lda,
          datatype* wr, datatype* wi, datatype* vl, const size_t ldvl, datatype* vr, const size_t ldvr);


template <typename datatype, typename computeEnv>
int syev(const SE_layout layout, char jobz, char uplo, const size_t n, double* a, const size_t lda, double* w);
}
