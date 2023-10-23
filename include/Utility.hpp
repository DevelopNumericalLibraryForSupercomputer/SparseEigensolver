#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <iterator>
#include <algorithm>
#include <array>
//#define MKL_Complex16 std::complex<double>
//#define MKL_Complex8  std::complex<float>

namespace SE{
//memory managament
template<typename datatype, computEnv comput_env>
datatype* malloc(const size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}

template<typename datatype, computEnv comput_env>
void free(datatype* ptr){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}

template<typename datatype, computEnv comput_env>
void memcpy(datatype* dest, const datatype* source, size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}

/* templated version of mkl wrapper.
 * Following functions and enum type variables are implemented
 * BLAS
 *  gemm
 *  daxpy
 *  SE_Transpose     // matrix transpose
 *  SE_layout        // matrix layout
 * 
 * LAPACK
 *  geev
 * 
 * FYI
enum CBLAS_LAYOUT {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
 */ 
//mkl - BLAS
enum SE_transpose{
    NoTrans,
    Trans,
    ConjTrans
};

enum SE_layout{
    RowMajor,
    ColMajor
};

//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
//output : C
template <typename datatype, computEnv comput_env>
void gemm(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb, const size_t m, const size_t n, const size_t k,
          const datatype alpha, const datatype *a, const size_t lda,
          const datatype *b, const size_t ldb, const datatype beta,
          datatype *c, const size_t ldc)
{  static_assert(false,"This is not implemented yet"); }

template <typename datatype, computEnv comput_env>
void axpy(const size_t n, const datatype a, const datatype *x, const size_t incx, datatype *y, const size_t incy)
{  static_assert(false,"This is not implemented yet"); }

template <typename datatype, computEnv comput_env>
void scal(const size_t n, const datatype alpha, datatype *x, const size_t incx)
{  static_assert(false,"This is not implemented yet"); }

//LAPACK
template <typename datatype, computEnv comput_env>
int geev(const SE_layout Layout, char jobvl, char jobvr, const size_t n, datatype* a, const size_t lda,
          datatype* wr, datatype* wi, datatype* vl, const size_t ldvl, datatype* vr, const size_t ldvr)
{  static_assert(false,"This is not implemented yet"); return -1;}

//numerical recipies
template <size_t dimension>
void cumprod(const std::array<size_t, dimension>& shape, std::array<size_t, dimension+1>& shape_mult, std::string indexing="F"){
    /* Ex1)
     * shape = {2, 3, 4}, indexing="F"
     * shape_mult = {1, 2, 6, 24}
     * Ex2)
     * shape = {2, 3, 4}, indexing="C"
     * shape_mult = {1, 4, 12, 24}
     */
    shape_mult[0] = 1;
    if (indexing == "F"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[i];
        }
    }
    else if(indexing == "C"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[dimension-i-1];
        }
    }
}



template<typename datatype>
std::vector<size_t> sort_indicies(const datatype* data_array, const size_t array_size){
    //std::array<size_t, array_size> idx;
    std::vector<size_t> idx;
    idx.resize(array_size);
    std::iota(std::begin(idx), std::end(idx), 0);
    std::stable_sort(std::begin(idx), std::end(idx), [data_array](size_t i1, size_t i2) {return data_array[i1] < data_array[i2];});
    return idx;
}

template <typename datatype, computEnv comput_env>
void eigenvec_sort(datatype* eigvals, datatype* eigvecs, const size_t n, const size_t lda)
{  static_assert(false,"This is not implemented yet");  }
}