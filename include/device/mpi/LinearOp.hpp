#pragma once
#include <complex>
#define MKL_Complex16 std::complex<double>
#include "Utility.hpp"
#include "Device.hpp"
#include "device/LinearOp.hpp"

#include "mkl.h"
#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"
namespace SE{
//typedef MKL_INT MDESC[ 9 ];
//const MKL_INT i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
//MKL_INT info;
//const double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;

//memory managament
template<>
double* malloc<double, DEVICETYPE::MPI>(const int size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}

template<>
void free<DEVICETYPE::MPI>(void* ptr) {
    std::free(ptr);
}


//ignore COPYTYPE: always NONE
template<>
void memcpy<int, DEVICETYPE::MPI>(int* dest, const int* source, int size, COPYTYPE copy_type){
    std::memcpy(dest, source, size * sizeof(int));
}

template <>
void scal<double, double, DEVICETYPE::MPI>(const int n, const double alpha, double *x, const int incx){
    return cblas_dscal(n, alpha, x, incx);
}
template <>
void scal<double,std::complex<double>, DEVICETYPE::MPI>(const int n, const double alpha, std::complex<double> *x, const int incx){
    return cblas_zdscal(n, alpha, x, incx);
}
template <>
void scal<std::complex<double>,std::complex<double>, DEVICETYPE::MPI>(const int n, const std::complex<double> alpha, std::complex<double> *x, const int incx){
    return cblas_zscal(n, &alpha, x, incx);
}




template <>
void axpy<double, DEVICETYPE::MPI>(const int n, const double a, const double *x, const int incx, double *y, const int incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}


///////////////////////////////////////////////////////////////////////////////////////////////// DEVICETYPE::MPI only 
template<>
void p_geadd<double>( const char* trans, 
              const int* m,
              const int* n,
              const double* alpha,
              const double* a,
              const int* ia, 
              const int* ja, 
              const int desca[9],
              const double* beta, 
                    double* c,
              const int* ic,
              const int* jc, 
              const int descc[9]){
    return pdgeadd(trans,m,n,alpha,a,ia,ja,desca,beta,c,ic,jc,descc);   
}

template<>
void p_gemm<double>( const char* trans1, 
             const char* trans2, 
             const int* m,
             const int* n,
             const int* k,
             const double* alpha,
             const double* a,
             const int* ia, 
             const int* ja, 
             const int desca[9],
             const double* b,
             const int* ib, 
             const int* jb, 
             const int descb[9],
             const double* beta, 
                   double* c,
             const int* ic,
             const int* jc, 
             const int descc[9]){
    return pdgemm(trans1,trans2, m,n,k,alpha,a,ia,ja,desca,b,ib,jb,descb,beta,c,ic,jc,descc);
}

template<>
void p_geqrf<double>( const int* m,
              const int* n,
                    double* a,
              const int* ia,
              const int* ja,
              const int desca[9],
                    double* tau,
                    double* work,
              const int* lwork,
                    int* info){
    return pdgeqrf(m,n,a,ia, ja,desca,tau, work, lwork, info);  
}

template<>
void p_orgqr<double>( const int* m,
              const int* n,
              const int* k,
                    double* a,
              const int* ia,
              const int* ja,
              const int desca[9],
              const double* tau,
                    double* work,
              const int* lwork, 
                    int* info
              ){
    return pdorgqr(m,n,k,a,ia,ja,desca,tau,work,lwork,info);
}
            

template<>
void gsum2d<double>( const int* icontxt,
                     const char* scope,
                     const char* top,
                     const int* m,
                     const int* n,
                           double* a,
                     const int* lda,
                     const int* rdest,
                     const int* cdest
             ){
    return dgsum2d(icontxt, scope, top, m,n,a,lda,rdest,cdest);  
}

template<>
void p_gemr2d<double>( const int* m,
               const int* n,
                     double* a,
               const int* ia,
               const int* ja,
               const int  desca[9],
                     double* b,
               const int* ib,
               const int* jb,
               const int  descb[9],
               const int* ictxt){
    return pdgemr2d(m,n,a,ia,ja,desca,b,ib,jb,descb,ictxt); 
}

template<>
void p_syevd<double>( const char* jobz,
              const char* uplo,
              const int* n,
                    double* a,
              const int* ia, 
              const int* ja, 
              const int  desca[9],
                    double* w,
                    double* z,
              const int* iz, 
              const int* jz, 
              const int  descz[9],
                    double* work,
              const int* lwork, 
                    double * rwork, 
              const int* lrwork, 
                    int* iwork, 
              const int* liwork,
                    int* info){
    return pdsyevd(jobz,uplo,n,a,ia,ja,desca,w,z,iz,jz,descz,work,lwork,iwork,liwork,info); 
}
///////////////////////////////////////////////////////////////////////////////////////////////// DEVICETYPE::MPI only 
template<>
void p_geadd<std::complex<double>>( const char* trans, 
              const int* m,
              const int* n,
              const std::complex<double>* alpha,
              const std::complex<double>* a,
              const int* ia, 
              const int* ja, 
              const int desca[9],
              const std::complex<double>* beta, 
                    std::complex<double>* c,
              const int* ic,
              const int* jc, 
              const int descc[9]){
    return pzgeadd(trans,m,n,alpha,a,ia,ja,desca, beta,c,ic,jc,descc);   
}

template<>
void p_gemm<std::complex<double>>( const char* trans1, 
             const char* trans2, 
             const int* m,
             const int* n,
             const int* k,
             const std::complex<double>* alpha,
             const std::complex<double>* a,
             const int* ia, 
             const int* ja, 
             const int desca[9],
             const std::complex<double>* b,
             const int* ib, 
             const int* jb, 
             const int descb[9],
             const std::complex<double>* beta, 
                   std::complex<double>* c,
             const int* ic,
             const int* jc, 
             const int descc[9]){
    return pzgemm(trans1,trans2,m,n,k,alpha,a,ia,ja,desca,b,ib,jb,descb,beta,c,ic,jc,descc);    
}

template<>
void p_geqrf<std::complex<double>>( const int* m,
              const int* n,
                    std::complex<double>* a,
              const int* ia,
              const int* ja,
              const int desca[9],
                    std::complex<double>* tau,
                    std::complex<double>* work,
              const int* lwork,
                    int* info){
    return pzgeqrf(m,n,a,ia,ja,desca, tau,work,lwork,info); 
}

template<>
void p_orgqr<std::complex<double>>( const int* m,
              const int* n,
              const int* k,
                    std::complex<double>* a,
              const int* ia,
              const int* ja,
              const int desca[9],
              const std::complex<double>* tau,
                    std::complex<double>* work,
              const int* lwork, 
                    int* info
              ){
    return pzungqr(m,n,k,a,ia,ja,desca,tau,work,lwork,info);    
}
            

template<>
void gsum2d<std::complex<double>>( const int* icontxt,
                     const char* scope,
                     const char* top,
                     const int* m,
                     const int* n,
                           std::complex<double>* a,
                     const int* lda,
                     const int* rdest,
                     const int* cdest
             ){
    return zgsum2d(icontxt, scope, top, m,n,reinterpret_cast<double*>(a),lda,rdest,cdest); 
}

template<>
void p_gemr2d<std::complex<double>>( const int* m,
               const int* n,
                     std::complex<double>* a,
               const int* ia,
               const int* ja,
               const int  desca[9],
                     std::complex<double>* b,
               const int* ib,
               const int* jb,
               const int  descb[9],
               const int* ictxt){
    return pzgemr2d(m,n,a,ia,ja,desca,b,ib,jb,descb,ictxt); 
}

template<>
void p_syevd<std::complex<double>>( const char* jobz,
              const char* uplo,
              const int* n,
                    std::complex<double>* a,
              const int* ia, 
              const int* ja, 
              const int  desca[9],
                    double* w,
                    std::complex<double>* z,
              const int* iz, 
              const int* jz, 
              const int  descz[9],
                    std::complex<double>* work,
              const int* lwork, 
                    double * rwork, 
              const int* lrwork, 
                    int* iwork, 
              const int* liwork,
                    int* info){
    return pzheevd(jobz,uplo,n,a,ia,ja,desca,w,z,iz,jz,descz,work,lwork,rwork,lrwork,iwork,liwork,info);    
}

//y =alpha*Ax +beta*y
/*
template <>
void sbmv<double, DEVICETPYE::MPI>(
            const char uplo, const int n, const int k,
            const double alpha,
            const double *a, const int lda,
            const double *x, const int incx,
            const double beta,
            double *y, const int incy
        ){
    return cblas_dsbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}
*/
/*
template <>
double nrm2<double, DEVICETYPE::MKL>(const int n, const double *x, const int incx){
    return pddnrm2(n, x, incx);
    assert (incx==0);
    MDESC   descA;
    work = (double*) mkl_calloc(mp, sizeof( double ), 64);
    descinit( descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
    assert (info==0);
    pdlange('f', &n, &i_one, x, &i_zero, &i_zero, descA, work  ) //frobius norm
    mkl_free(work);
    return cblas_dnrm2(n, x, incx);
}

template <>
void gemv<double, DEVICETYPE::MPI>(const ORDERTYPE order, const TRANSTYPE transa, const int m, const int n, const double alpha,
                       const double *a, const int lda, const double *x, const int incx,
                       const double beta, double *y, const int incy)
{
    return cblas_dgemv(map_order_blas_MPI(order), map_transpose_blas_MPI(transa), m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template<>
void gemm<double, DEVICETYPE::MPI>(const ORDERTYPE order, const TRANSTYPE transa, const TRANSTYPE transb,
                       const int m, const int n, const int k,
                       const double alpha, const double *a, const int lda,
                       const double *b, const int ldb, const double beta,
                       double *c, const int ldc){
    return cblas_dgemm(map_order_blas_MPI(order), map_transpose_blas_MPI(transa), map_transpose_blas_MPI(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}


//template<>
//int geqrf<double, DEVICETYPE::MPI>(const LAYOUT layout, int m, int n, double* a, int lda, double* tau){
//    return LAPACKE_dgeqrf(map_layout_lapack_MPI(layout), m, n, a, lda, tau);
//}
//
//template<>
//int orgqr<double, DEVICETYPE::MPI>(const LAYOUT layout, int m, int n, double* a, int lda, double* tau){
//    return LAPACKE_dorgqr(map_layout_lapack_MPI(layout), m, m, n, a, lda, tau);
//}
//
//template <>
//int geev<double, DEVICETYPE::MPI>(const LAYOUT layout, char jobvl, char jobvr, const int n, double* a, const int lda,
//          double* wr, double* wi, double* vl, const int ldvl, double* vr, const int ldvr){
//    return LAPACKE_dgeev(map_layout_lapack_MPI(layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
//}
//template <>
//int syev<double, DEVICETYPE::MPI>(const LAYOUT layout, char jobz, char uplo, const int n, double* a, const int lda, double* w){
//    return LAPACKE_dsyev(map_layout_lapack_MPI(layout), jobz, uplo, n, a, lda, w);
//}
*/

}
