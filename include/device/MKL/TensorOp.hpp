#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "../../DenseTensor.hpp"
#include "../../SparseTensor.hpp"
#include "MKLComm.hpp"

namespace SE{
//spmv
template <typename maptype1, typename maptype2>
Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* spmv(Tensor<STORETYPE::Dense, double, 2, MKL, maptype1>* a, 
                                                        Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<size_t,1> return_size = {m};
    double* return_data = malloc<double, MKL>(m);

    gemm<double, MKL>(SE_layout::ColMajor, transa, SE_transpose::NoTrans, m, 1, k, 1.0, a->data, m, v->data, k, 0.0, return_data, m);
    maptype2* return_map = new maptype2(return_size, 1);
    Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>(v->comm, return_map, return_size, return_data);
    return return_mat;
}   

template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* spmv(Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* v,
                                                        SE_transpose transa){
    return matmul(a, v, transa, SE_transpose::NoTrans);
}   

template <typename maptype1, typename maptype2>
Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* spmv(Tensor<STORETYPE::COO, double, 2, MKL, maptype1>* a, 
                                                        Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<size_t,1> return_size = {m};
    double* return_data = malloc<double, MKL>(m);
    memset<double, MKL>(return_data, 0, m);
    
    if(transa == SE_transpose::NoTrans){
        for(auto entity : a->data){
            return_data[ entity.first[0] ] += entity.second * v->data[ entity.first[1] ];
        }
    }
    else{
        for(auto entity : a->data){
            return_data[ entity.first[1] ] += entity.second * v->data[ entity.first[0] ];
        }
    }
    maptype2* return_map = new maptype2(return_size, 1);
    Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, MKL, maptype2>(a->comm, return_map, return_size, return_data);
    return return_mat;
}   
template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* spmv(Tensor<STORETYPE::COO, double, 2, MKL, maptype>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    size_t number_of_vec = v->shape[1];
    std::array<size_t,2> return_size = {m, number_of_vec};
    double* return_data = malloc<double, MKL>(m*number_of_vec);
    memset<double, MKL>(return_data, 0, m*number_of_vec);
    
    if(transa == SE_transpose::NoTrans){
        for(auto entity : a->data){
            for(int n = 0; n<number_of_vec ; n++){
                return_data[ entity.first[0] + n*m ] += entity.second * v->data[ entity.first[1] + n*m];
            }
        }
    }
    else{
        for(auto entity : a->data){
            for(int n = 0; n<number_of_vec ; n++){
                return_data[ entity.first[1] + n*m ] += entity.second * v->data[ entity.first[1] + n*m];
            }
        }
    }
    maptype* return_map = new maptype(return_size, 1);
    Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* return_mat = new Tensor<STORETYPE::Dense, double, 2, MKL, maptype>(a->comm, return_map, return_size, return_data);
    return return_mat;
}   

//matmul
//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* matmul(Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* a,
                                                          Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* b,
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
    double* return_data = malloc<double, MKL>(m*n);

    gemm<double, MKL>(SE_layout::ColMajor, transa, transb, m, n, k, 1.0, a->data, m, b->data, k, 0.0, return_data, m);
    maptype* return_map = new maptype(return_size, 1);
    Tensor<STORETYPE::Dense, double, 2, MKL, maptype>* return_mat = new Tensor<STORETYPE::Dense, double, 2, MKL, maptype>(a->comm, return_map, return_size, return_data);
    return return_mat;
}
//QR
template <>
void orthonormalize<double, MKL>(double* eigvec, size_t vector_size, size_t number_of_vectors, std::string method)
{
    if(method == "qr"){
        double* tau = malloc<double, MKL>(number_of_vectors);
        int info = geqrf<double, MKL>(SE_layout::ColMajor, vector_size, number_of_vectors, eigvec, vector_size, tau);
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        info = orgqr<double, MKL>(SE_layout::ColMajor, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        free<double, MKL>(tau);
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        double* submatrix = malloc<double, MKL>(number_of_vectors*number_of_vectors);
        double* submatrix_eigvals = malloc<double, MKL>(number_of_vectors);
        gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, number_of_vectors, number_of_vectors, vector_size, 1.0, eigvec, number_of_vectors, eigvec, vector_size, 0.0, submatrix, number_of_vectors);
        syev<double, MKL>(SE_layout::ColMajor, 'V', 'U', number_of_vectors, submatrix, number_of_vectors, submatrix_eigvals);
        double* new_eigvec = malloc<double, MKL>(vector_size*number_of_vectors);
        gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans,vector_size, number_of_vectors, number_of_vectors, 1.0, eigvec, vector_size, submatrix, number_of_vectors, 0.0, new_eigvec, vector_size);
        memcpy<double, MKL>(eigvec, new_eigvec, vector_size*number_of_vectors);
        free<double, MKL>(submatrix);
        free<double, MKL>(submatrix_eigvals);
        free<double, MKL>(new_eigvec);
    }
}
}
