#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "../../DenseTensor.hpp"
#include "../../SparseTensor.hpp"

namespace SE{
//spmv
template <>
Tensor<STORETYPE::Dense, double, 2, MPI, maptype>* spmv(Tensor<STORETYPE::COO, double, 2, MPI, maptype>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, MPI, maptype>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[1] == k);

    std::array<size_t,1> return_size = {m};
    double* return_data = malloc<double, MPI>(m);
    memset<double, MPI>(return_data, 0, m);
    
    if(transa == SE_transpose::NoTrans){
        for(auto entity : a->data){
            return_data[ entity.first[0] ] += entity.second * v[ entity.first[1] ];
        }
    }
    else{
        for(auto entity : a->data){
            return_data[ entity.first[1] ] += entity.second * v[ entity.first[0] ];
        }
    }

    DenseTensor<double, 1, MPI, maptype>* return_mat = new DenseTensor<double, 1, MPI, maptype>(return_size, return_data);
    return return_mat;
}   


//matmul
//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, MPI, maptype>* matmul(Tensor<STORETYPE::Dense, double, 2, MPI, maptype>* a,
                                                          Tensor<STORETYPE::Dense, double, 2, MPI, maptype>* b,
                                                          SE_transpose transa = SE_transpose::NoTrans,
                                                          SE_transpose transb = SE_transpose::NoTrans){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    size_t n = b->shape[1];
    if(transb != SE_transpose::NoTrans){
        n = b->shape[0];
        assert(k == b->shape[1]);    
    }
    else{
        assert(k == b->shape[0]);
    }
    std::array<size_t,2> return_size = {m,n};
    double* return_data = malloc<double, MPI>(m*n);

    gemm<double, MPI>(SE_layout::ColMajor, transa, transb, m, n, k, 1.0, a->data, m, b->data, k, 0.0, return_data, m);
    
    DenseTensor<double, 2, MPI, maptype>* return_mat = new DenseTensor<double, 2, MPI, maptype>(return_size, return_data);
    return return_mat;
}

//QR
template <>
void orthonormalize<double, MPI>(double* eigvec, size_t vector_size, size_t number_of_vectors, std::string method)
{
    if(method == "qr"){
        std::cout << "qr decomposition for MPI parallelization is not available" << std::endl;
        exit(-1);
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        double* submatrix = malloc<double, MPI>(number_of_vectors*number_of_vectors);
        double* submatrix_eigvals = malloc<double, MPI>(number_of_vectors);
        gemm<double, MPI>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, number_of_vectors, number_of_vectors, vetor_size, 1.0, eigvec, number_of_vectors, eigvec, vector_size, 0.0, submatrix, number_of_vectors);
        syev<double, MPI>(SE_layout::ColMajor, 'V', 'U', number_of_vectors, submatrix, number_of_vectors, submatrix_eigvals);
        double* new_eigvec = malloc<double, MPI>(vector_size*number_of_vectors);
        gemm<double, MPI>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans,vector_size, number_of_vectors, number_of_vectors, 1.0, eigvec, vector_size, submatrix, number_of_vectors, 0.0, new_eigvec, vector_size);
        memcpy<double, MPI>(eigvec, new_eigvec, vector_size*number_of_vectors);
        free<double, MPI>(submatrix);
        free<double, MPI>(submatrix_eigvals);
        free<double, MPI>(new_eigvec);
    }
}



}
