#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MPIComm.hpp"

namespace SE{
//spmv
template <typename maptype1, typename maptype2>
Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* spmv(Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype1>* a, 
                                                         Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* v,
                                                         SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<size_t,1> return_size = {m};

    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k);
        size_t* recvcounts = v->map.get_partition_size_array();
        v->comm->allgatherv(v->data, recvcounts[v->comm->rank], vector, recvcounts);
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size);
                gemm<double, SEMpi>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, my_size, 1, k, 1.0, a->data, my_size, vector, k, 0.0, return_data, my_size);
                //maptype2* return_map = new maptype2(return_size, a->comm->world_size, 0);
                Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(a->comm, return_size, return_data);
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m);
            gemm<double, SEMpi>(SE_layout::ColMajor, transa, SE_transpose::NoTrans, m, 1, k, 1.0, a->data, m, vector, k, 0.0, return_data, m);
            //maptype2* return_map = new maptype2(return_size, a->comm->world_size);
            Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(v->comm, return_size, return_data);
            return return_mat;
        }
    }
    else{
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size);
                
                gemm<double, SEMpi>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, my_size, 1, k, 1.0, a->data, my_size, v->data, k, 0.0, return_data, my_size);
                
                //maptype2* return_map = new maptype2(return_size, a->comm->world_size, 0);
                Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(a->comm, return_size, return_data, true, 0);
                
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m);
            gemm<double, SEMpi>(SE_layout::ColMajor, transa, SE_transpose::NoTrans, m, 1, k, 1.0, a->data, m, v->data, k, 0.0, return_data, m);
            //maptype2* return_map = new maptype2(return_size, a->comm->world_size);
            Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(v->comm, return_size, return_data);
            return return_mat;
        }
    }

}   

template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* spmv(Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* v,
                                                        SE_transpose transa){
    return matmul(a, v, transa, SE_transpose::NoTrans);
}   

template <typename maptype1, typename maptype2>
Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* spmv(Tensor<STORETYPE::COO, double, 2, SEMpi, maptype1>* a, 
                                                        Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<size_t,1> return_size = {m};


    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k);
        size_t* recvcounts = v->map.get_partition_size_array();
        v->comm->allgatherv(v->data, recvcounts[v->comm->rank], vector, recvcounts);
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size);
                memset<double, SEMpi>(return_data, 0, my_size);
                
                for(auto entity : a->data){
                    return_data[ a->map.get_local_index(entity.first[0], rank) ] += entity.second * vector[ entity.first[1] ];
                }
                
                //maptype2* return_map = new maptype2(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(a->comm, return_size, return_data, true, 0);
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m);
            memset<double, SEMpi>(return_data, 0, m);
            
            if(transa == SE_transpose::NoTrans){
                for(auto entity : a->data){
                    return_data[ entity.first[0] ] += entity.second * vector[ entity.first[1] ];
                }
            }
            else{
                for(auto entity : a->data){
                    return_data[ entity.first[1] ] += entity.second * vector[ entity.first[0] ];
                }
            }
            
            //maptype2* return_map = new maptype2(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(a->comm, return_size, return_data);
            return return_mat;
        }
    }
    else{
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size);
                memset<double, SEMpi>(return_data, 0, my_size);
                a->comm->barrier();
                for(auto entity : a->data){
                    return_data[ a->map.get_local_index(entity.first[0], rank) ] += entity.second * v->data[ entity.first[1] ];
                }
                a->comm->barrier();
                //maptype2* return_map = new maptype2(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(a->comm, return_size, return_data, true, 0);
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m);
            memset<double, SEMpi>(return_data, 0, m);
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
            //maptype2* return_map = new maptype2(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, maptype2>(v->comm, return_size, return_data);
            return return_mat;
        }
    }
}   

template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* spmv(Tensor<STORETYPE::COO, double, 2, SEMpi, maptype>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* v,
                                                        SE_transpose transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    size_t number_of_vec = v->shape[1];
    std::array<size_t,2> return_size = {m, number_of_vec};

    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k*number_of_vec);
        size_t* recvcounts = v->map.get_partition_size_array();
        for(int n=0; n<number_of_vec ; n++){
            v->comm->allgatherv(&v->data[n*recvcounts[v->comm->rank]], recvcounts[v->comm->rank], &vector[n*k], recvcounts);
        }
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size*number_of_vec);
                memset<double, SEMpi>(return_data, 0, my_size*number_of_vec);

                for(auto entity : a->data){
                    for(int n=0; n<number_of_vec ; n++){
                        return_data[ a->map.get_local_index(entity.first[0], rank) + n*my_size] += entity.second * vector[ entity.first[1] + n*my_size];
                    }
                }

                //maptype* return_map = new maptype(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>(a->comm, return_size, return_data, true, 0);
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m*number_of_vec);
            memset<double, SEMpi>(return_data, 0, m*number_of_vec);

            if(transa == SE_transpose::NoTrans){
                for(auto entity : a->data){
                    for(int n = 0; n<number_of_vec ; n++){
                        return_data[ entity.first[0] + n*m ] += entity.second * vector[ entity.first[1] + n*m];
                    }
                }
            }
            else{
                for(auto entity : a->data){
                    for(int n = 0; n<number_of_vec ; n++){
                        return_data[ entity.first[1] + n*m ] += entity.second * vector[ entity.first[1] + n*m];
                    }
                }
            }            

            //maptype* return_map = new maptype(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>(a->comm, return_size, return_data);
            return return_mat;
        }
    }
    else{
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                size_t rank = a->comm->rank;
                size_t my_size = a->map.get_my_partition_size(rank);
                double* return_data = malloc<double, SEMpi>(my_size*number_of_vec);
                memset<double, SEMpi>(return_data, 0, my_size*number_of_vec);

                for(auto entity : a->data){
                    for(int n=0; n<number_of_vec ; n++){
                        return_data[  a->map.get_local_index(entity.first[0], rank)  + n*my_size] += entity.second * v->data[ entity.first[1] + n*my_size];
                    }
                }
                //maptype* return_map = new maptype(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>(a->comm, return_size, return_data, true, 0);
                return return_mat;
            }
            else{
                std::cout << "to be implemented" << std::endl;
                exit(-1);
            }
        }
        else{
            double* return_data = malloc<double, SEMpi>(m*number_of_vec);
            memset<double, SEMpi>(return_data, 0, m*number_of_vec);

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
            //maptype* return_map = new maptype(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>(a->comm, return_size, return_data);
            return return_mat;
        }
    }
}


//matmul
//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
template <typename maptype>
Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* matmul(Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* a,
                                                          Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* b,
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
    if( (!a->map.is_sliced) && (!b->map.is_sliced)){
        double* return_data = malloc<double, SEMpi>(m*n);

        gemm<double, SEMpi>(SE_layout::ColMajor, transa, transb, m, n, k, 1.0, a->data, m, b->data, k, 0.0, return_data, m);
        //maptype* return_map = new maptype(return_size, a->comm->world_size);
        Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>* return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, maptype>(a->comm, return_size, return_data);
        return return_mat;
    }
    else{
        std::cout << "to be implemented" << std::endl;
        exit(-1);
    }
}

//QR
template <>
void orthonormalize<double, SEMpi>(double* eigvec, size_t vector_size, size_t number_of_vectors, std::string method)
{
    if(method == "qr"){
        std::cout << "qr decomposition for MPI parallelization is not available" << std::endl;
        exit(-1);
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        double* submatrix = malloc<double, SEMpi>(number_of_vectors*number_of_vectors);
        double* submatrix_eigvals = malloc<double, SEMpi>(number_of_vectors);
        gemm<double, SEMpi>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, number_of_vectors, number_of_vectors, vector_size, 1.0, eigvec, number_of_vectors, eigvec, vector_size, 0.0, submatrix, number_of_vectors);
        syev<double, SEMpi>(SE_layout::ColMajor, 'V', 'U', number_of_vectors, submatrix, number_of_vectors, submatrix_eigvals);
        double* new_eigvec = malloc<double, SEMpi>(vector_size*number_of_vectors);
        gemm<double, SEMpi>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans,vector_size, number_of_vectors, number_of_vectors, 1.0, eigvec, vector_size, submatrix, number_of_vectors, 0.0, new_eigvec, vector_size);
        memcpy<double, SEMpi>(eigvec, new_eigvec, vector_size*number_of_vectors);
        free<double, SEMpi>(submatrix);
        free<double, SEMpi>(submatrix_eigvals);
        free<double, SEMpi>(new_eigvec);
    }
}



}
