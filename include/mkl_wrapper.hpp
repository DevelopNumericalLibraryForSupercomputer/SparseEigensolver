#pragma once
#include "Device.hpp"
/* templated version of mkl wrapper.
 * Containing functions
 * BLAS
 *  gemm
 *  daxpy
 */ 

namespace TH{
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
void gemm(const TH_layout Layout, const TH_transpose transa, const TH_transpose transb,
          const size_t m, const size_t n, const size_t k,
          const datatype alpha, const datatype *a, const size_t lda,
          const datatype *b, const size_t ldb, const datatype beta,
          datatype *c, const size_t ldc)
{  static_assert(false,"This is not implemented yet"); }

template <typename datatype, typename device>
void axpy(const size_t n, const datatype a, const datatype *x, const size_t incx, datatype *y, const size_t incy)
{  static_assert(false,"This is not implemented yet"); }

}