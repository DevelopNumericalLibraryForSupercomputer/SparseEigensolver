#pragma once
//#include <iostream>
//#include <cstdlib>
#include <cstring>
#include <iostream>
#include <cassert>
//#include <numeric>
//#include <iterator>
//#include <algorithm>
//#include <array>

#include "Type.hpp"
#include "Utility.hpp"

namespace SE{

//memory managament
template<typename DATATYPE, DEVICETYPE device =DEVICETYPE::BASE>
DATATYPE* malloc(const int size){
    return static_cast<DATATYPE*>(std::malloc(size * sizeof(DATATYPE)));
}

//template<>
//double* malloc<double, DEVICETYPE::MKL>(const int size);
//template<>
//double* malloc<double, DEVICETYPE::MPI>(const int size);

template< DEVICETYPE device=DEVICETYPE::BASE>
void free(void* ptr){
    std::free(ptr);
}

//template<>
//void free <DEVICETYPE::MKL>(void* ptr);
//template<>
//void free <DEVICETYPE::MPI>(void* ptr);

template<typename DATATYPE, DEVICETYPE device=DEVICETYPE::BASE>
void memcpy(DATATYPE* dest, const DATATYPE* source, int size, COPYTYPE copy_type=COPYTYPE::NONE){
//    std::cout << typeid(copy_type).name() <<"\t"<< typeid(COPYTYPE::NONE).name() <<std::endl;
//    std::cout << (int) copy_type <<"\t" << (int) COPYTYPE::NONE <<std::endl;
    assert( (COPYTYPE::NONE==(COPYTYPE) copy_type) or (COPYTYPE::DEVICE2DEVICE==(COPYTYPE) copy_type) );
  
    std::memcpy(dest, source, size * sizeof(DATATYPE));
}

template<typename DATATYPE, DEVICETYPE device=DEVICETYPE::BASE>
void memset(DATATYPE* dest, const int value, const int size){
    std::memset(dest, value, size * sizeof(DATATYPE));
}


//x = a * x
template <typename DATATYPE1, typename DATATYPE2, DEVICETYPE device>
void scal(const int n, const DATATYPE1 alpha, DATATYPE2 *x, const int incx);

//a * x + y
template <typename DATATYPE, DEVICETYPE device>
void axpy(const int n, const DATATYPE a, const DATATYPE *x, const int incx, DATATYPE *y, const int incy);

//y := alpha*A*x + beta*y,
template <typename DATATYPE, DEVICETYPE device>
void sbmv(const ORDERTYPE layout, 
            const char uplo, const int n, const int k,
            const DATATYPE alpha,
            const DATATYPE *a, const int lda,
            const DATATYPE *x, const int incx,
            const DATATYPE beta,
            DATATYPE *y, const int incy
	    );

//Euclidean norm, ||x||
template <typename DATATYPE, DEVICETYPE device>
typename real_type<DATATYPE>::type nrm2(const int n, const DATATYPE *x, const int incx);

//y (i*N+incy) = x (i*M+incx) 
template <typename DATATYPE, DEVICETYPE device>
void copy(const int n, const DATATYPE *x, const int incx, DATATYPE *y, const int incy);

//alpha * A * x + beta * y
template <typename DATATYPE, DEVICETYPE device>
void gemv(const ORDERTYPE layout, const TRANSTYPE transa, const int m, const int n, const DATATYPE alpha,
          const DATATYPE *a, const int lda, const DATATYPE *x, const int incx,
          const DATATYPE beta, DATATYPE *y, const int incy);

//alpha * A * x + b * y
//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
//output : C
template <typename DATATYPE, DEVICETYPE device>
void gemm(const ORDERTYPE layout, const TRANSTYPE transa, const TRANSTYPE transb, const int m, const int n, const int k,
          const DATATYPE alpha, const DATATYPE *a, const int lda,
          const DATATYPE *b, const int ldb, const DATATYPE beta,
          DATATYPE *c, const int ldc);

//Blas-like extensions
//void mkl_domatcopy (char ordering, char trans, int rows, int cols,
//                   const double alpha, const double * A, int lda, double * B, int ldb);
template <typename DATATYPE, DEVICETYPE device>
void omatcopy(const ORDERTYPE layout, const TRANSTYPE trans, int rows, int cols,
              const double alpha, DATATYPE *a, int lda, DATATYPE *b, int ldb);


//LAPACK

//QR decomposition of general (possible negative diagonal) m by n matrix, without pivoting
//for the substentiall tall matrix, see geqr
//If we don't need to directly apply q matrix and just need to multiply with othter matrix, see ormqr
template <typename DATATYPE, DEVICETYPE device>
int geqrf(const ORDERTYPE layout, int m, int n, DATATYPE* a, int lda, DATATYPE* tau);

//generate real orthogonal matrix Q from geqrf
//LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
//lapack_int LAPACKE_dorgqr( int matrix_layout, lapack_int m, lapack_int n,
//                           lapack_int k, double* a, lapack_int lda,
//                           const double* tau );
template <typename DATATYPE, DEVICETYPE device>
int orgqr(const ORDERTYPE layout, int m, int n, DATATYPE* a, int lda, DATATYPE* tau);

//ormqr

//diagonalization of general n by n matrix
//lapack_int LAPACKE_dgeev( int matrix_layout, char jobvl, char jobvr,
                          //lapack_int n, double* a, lapack_int lda, double* wr,
                          //double* wi, double* vl, lapack_int ldvl, double* vr,
//                          lapack_int ldvr );
//

// The arguments of geev function are not trivial.
// for dgeev function, wr and wi are required 
// but for zgeev function, only w is required
// Thus, this wrapper function takes complex<double> as w regardless of datatype
// 
template <typename DATATYPE, DEVICETYPE device>
int geev(const ORDERTYPE layout, const char jobvl, const char jobvr, const int n, DATATYPE* a, const int lda,
          std::complex<typename real_type<DATATYPE>::type>* w, DATATYPE* vl, const int ldvl, DATATYPE* vr, const int ldvr);


template <typename DATATYPE, DEVICETYPE device>
int syev(const ORDERTYPE layout, const char jobz, const char uplo, const int n, DATATYPE* a, const int lda, typename real_type<DATATYPE>::type* w);

template <typename DATATYPE, DEVICETYPE device>
void vMul(const int n, const DATATYPE* a, const DATATYPE* b, DATATYPE* y );

///////////////////////////////////////////////////////////////////////////////////////////////// DEVICETYPE::MPI only 
template<typename DATATYPE>
void p_geadd( const char* trans, 
              const int* m,
              const int* n,
			  const DATATYPE* alpha,
			  const DATATYPE* a,
			  const int* ia, 
			  const int* ja, 
   		      const int desca[9],
			  const DATATYPE* beta, 
			        DATATYPE* c,
			  const int* ic,
			  const int* jc, 
   		      const int descc[9]);

template<typename DATATYPE>
void p_gemm( const char* trans1, 
             const char* trans2, 
			 const int* m,
			 const int* n,
			 const int* k,
			 const DATATYPE* alpha,
			 const DATATYPE* a,
			 const int* ia, 
			 const int* ja, 
   		     const int desca[9],
			 const DATATYPE* b,
			 const int* ib, 
			 const int* jb, 
   		     const int descb[9],
			 const DATATYPE* beta, 
			       DATATYPE* c,
			 const int* ic,
			 const int* jc, 
   		     const int descc[9]);

template<typename DATATYPE>
void p_geqrf( const int* m,
			  const int* n,
			        DATATYPE* a,
			  const int* ia,
			  const int* ja,
			  const int desca[9],
			        DATATYPE* tau,
			        DATATYPE* work,
			  const int* lwork,
			        int* info);

template<typename DATATYPE>
void p_orgqr( const int* m,
              const int* n,
			  const int* k,
			        DATATYPE* a,
			  const int* ia,
			  const int* ja,
			  const int desca[9],
			  const DATATYPE* tau,
			        DATATYPE* work,
			  const int* lwork, 
			        int* info
			  );
			

template<typename DATATYPE>
void gsum2d( const int* icontxt,
			 const char* scope,
			 const char* top,
			 const int* m,
			 const int* n,
				   DATATYPE* a,
			 const int* lda,
			 const int* rdest,
			 const int* cdest
             );

template<typename DATATYPE>
void p_gemr2d( const int* m,
               const int* n,
			         DATATYPE* a,
			   const int* ia,
			   const int* ja,
			   const int  desca[9],
			         DATATYPE* b,
			   const int* ib,
			   const int* jb,
			   const int  descb[9],
			   const int* ictxt);

template<typename DATATYPE>
void p_syevd( const char* jobz,
			  const char* uplo,
			  const int* n,
			        DATATYPE* a,
			  const int* ia, 
			  const int* ja, 
			  const int  desca[9],
				    typename real_type<DATATYPE>::type* w,
					DATATYPE* z,
			  const int* iz, 
			  const int* jz, 
			  const int  descz[9],
			        DATATYPE* work,
			  const int* lwork,
			        typename real_type<DATATYPE>::type* rwork,
			  const int* lrwork,
			        int* iwork, 
			  const int* liwork,
			        int* info);
              
}
