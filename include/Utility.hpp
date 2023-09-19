//ChatGPT3.5
#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "Device.hpp"

namespace TH{
//memory managament
template<typename datatype, typename device> datatype* malloc(const size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}
template<typename datatype, typename device> void free(datatype* ptr){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}
template<typename datatype, typename device> void memcpy(datatype* dest, const datatype* source, size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}

/* templated version of mkl wrapper.
 * Following functions and enum type variables are implemented
 * BLAS
 *  gemm
 *  daxpy
 *  TH_Transpose     // matrix transpose
 *  TH_layout        // matrix layout
 */ 
enum TH_transpose{
    Blas_NoTrans,
    Blas_Trans,
    Blas_ConjTrans
};

enum TH_layout{
    Blas_RowMajor,
    Blas_ColMajor
};

template <typename datatype, typename device>
void gemm(const TH_layout Layout, const TH_transpose transa, const TH_transpose transb, const size_t m, const size_t n, const size_t k,
          const datatype alpha, const datatype *a, const size_t lda,
          const datatype *b, const size_t ldb, const datatype beta,
          datatype *c, const size_t ldc)
{  static_assert(false,"This is not implemented yet"); }

template <typename datatype, typename device>
void axpy(const size_t n, const datatype a, const datatype *x, const size_t incx, datatype *y, const size_t incy)
{  static_assert(false,"This is not implemented yet"); }

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

}