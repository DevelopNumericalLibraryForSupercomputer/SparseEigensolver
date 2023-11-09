#pragma once
#include "../../Utility.hpp"
#include "LinearOp.hpp"
#include "mkl.h"

namespace SE{
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
