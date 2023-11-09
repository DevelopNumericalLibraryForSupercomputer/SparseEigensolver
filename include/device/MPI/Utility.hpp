#pragma once
#include "../../Utility.hpp"


namespace SE{

double* malloc<double, MPI>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}

template<>
void free<double, MPI>(double* ptr) {
    std::free(ptr);
}

template<>
void memcpy<double, MPI>(double* dest, const double* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

template<>
void memset<double, MPI>(double* dest, int value, size_t size){
    std::memset(dest, value, size * sizeof(double));
}


CBLAS_LAYOUT map_layout_blas(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return CblasRowMajor;
        case SE_layout::ColMajor: return CblasColMajor;
    }
    exit(-1);
}

int map_layout_lapack(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return LAPACK_ROW_MAJOR;
        case SE_layout::ColMajor: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose(SE_transpose trans){
    switch (trans){
        case SE_transpose::NoTrans:   return CblasNoTrans;
        case SE_transpose::Trans:     return CblasTrans;
        case SE_transpose::ConjTrans: return CblasConjTrans;
    }
    exit(-1);
}

template<>
void gemm<double, MPI>(const SE_layout Layout, const SE_transpose transa, const SE_transpose transb,
                       const size_t m, const size_t n, const size_t k,
                       const double alpha, const double *a, const size_t lda,
                       const double *b, const size_t ldb, const double beta,
                       double *c, const size_t ldc){
    return cblas_dgemm(map_layout_blas(Layout), map_transpose(transa), map_transpose(transb), m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

template <>
void scal<double, MPI>(const size_t n, const double alpha, double *x, const size_t incx){
    return cblas_dscal(n, alpha, x, incx);
}

template <>
void axpy<double, MPI>(const size_t n, const double a, const double *x, const size_t incx, double *y, const size_t incy){
    return cblas_daxpy(n, a, x, incx, y, incy);
}

template <>
int geev<double, MPI>(const SE_layout Layout, char jobvl, char jobvr, const size_t n, double* a, const size_t lda,
          double* wr, double* wi, double* vl, const size_t ldvl, double* vr, const size_t ldvr){
    return LAPACKE_dgeev(map_layout_lapack(Layout), jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr);
}

/*
template <>
void eigenvec_sort<double, MKL>(double* eigvals, double* eigvecs, const size_t number_of_eigvals, const size_t vector_size){
    double* new_eigvals = new double[number_of_eigvals];
    double* new_eigvecs = new double[number_of_eigvals*vector_size];
    std::vector<size_t> sorted_indicies = sort_indicies<double>(eigvals, number_of_eigvals);
    for(int i=0;i<number_of_eigvals;i++){
        new_eigvals[i] = eigvals[sorted_indicies[i]];
        for(int j=0;j<vector_size;j++){
            new_eigvecs[i*number_of_eigvals+j] = eigvecs[sorted_indicies[i]*number_of_eigvals+j];
        }
    }
    
    memcpy<double, MKL>(eigvals, new_eigvals, number_of_eigvals);
    memcpy<double, MKL>(eigvecs, new_eigvecs, number_of_eigvals*vector_size);
}

template <>
void orthonormalize<double, MKL>(double* eigvec, size_t vector_size, size_t number_of_vectors, std::string method)
{
    if(method == "qr"){
        double* tau = malloc<double, MKL>(number_of_vectors);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, vector_size, number_of_vectors, eigvec, vector_size, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(-1);
    }
}


*/


}