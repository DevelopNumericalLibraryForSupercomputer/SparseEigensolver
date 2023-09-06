#include <complex>
#define MKL_Complex16 std::complex<double>
//#define MKL_Complex8  std::complex<float>
#include "mkl.h"

/* templated version of mkl wrapper. Work only for std::complex<double> and double types.
 *
 * Containing functions
 *
 * BLAS
 *  gemm
 *  daxpy
 */ 


namespace TensorHetero{
template <typename datatype>
void gemm(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
          const MKL_INT m, const MKL_INT n, const MKL_INT k,
          const datatype alpha, const datatype *a, const MKL_INT lda,
          const datatype *b, const MKL_INT ldb, const datatype beta,
          datatype *c, const MKL_INT ldc)
{  static_assert(false,"This is not implemented yet"); }

template<>
void gemm<double>(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, 
                  const MKL_INT m, const MKL_INT n, const MKL_INT k, 
                  const double alpha, const double *a, const MKL_INT lda, 
                  const double *b, const MKL_INT ldb, const double beta, 
                  double *c, const MKL_INT ldc)
{ return cblas_dgemm(Layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); }

template<>
void gemm<std::complex<double> >(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                                 const MKL_INT m, const MKL_INT n, const MKL_INT k,
                                 const std::complex<double> alpha, const std::complex<double> *a, const MKL_INT lda,
                                 const std::complex<double> *b, const MKL_INT ldb, const std::complex<double> beta,
                                 std::complex<double> *c, const MKL_INT ldc)
{ return cblas_zgemm(Layout, transa, transb, m, n, k, (const void*) &alpha, (const void*) a, lda, (const void*) b, ldb, (const void*) &beta, (void*) c, ldc); }

template <typename datatype>
void axpy(const MKL_INT n, const datatype a, const datatype *x, const MKL_INT incx, datatype *y, const MKL_INT incy)
{  static_assert(false,"This is not implemented yet"); }

template<>
void axpy<double>(const MKL_INT n, const double a, const double *x, const MKL_INT incx, double *y, const MKL_INT incy)
{ return cblas_daxpy(n, a, x, incx, y, incy); }

}