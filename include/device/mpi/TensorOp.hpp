#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MPIComm.hpp"
#include "../../Gather.hpp"
namespace SE{
// function declaration is in device/TensorOp.hpp

//dense mv 
template <typename DATATYPE>
DenseTensor<1,DATATYPE,Contiguous1DMap<1>, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, DEVICETYPE::MPI>& mat,
    const DenseTensor<1, DATATYPE, Contiguous1DMap<1>, DEVICETYPE::MPI>& vec,
    TRANSTYPE trans=TRANSTYPE::N)
{
    auto world_size = mat.comm.get_world_size();
    auto rank = mat.comm.get_rank();

    assert (world_size == vec.comm.get_world_size());
    assert (rank == vec.comm.get_rank());

    size_t contract_dim = (trans==TRANSTYPE::N)? 1:0;
    size_t remained_dim = (trans==TRANSTYPE::N)? 0:1;
    
    assert (  mat.map.get_global_shape(contract_dim) == vec.map.get_global_shape(0) );
    std::cout << "=============" << std::endl;
    std::cout << "contract dim  = " << contract_dim << std::endl;
    std::cout << "mat.global_shape(0) = " << mat.map.get_global_shape(0) << std::endl;
    std::cout << "mat.global_shape(1) = " << mat.map.get_global_shape(1) << std::endl;
    std::cout << "vec.global_shape(0) = " << vec.map.get_global_shape(0) << std::endl;
    std::cout << "=============" << std::endl;
    
    std::array<size_t, 1> output_shape = {mat.map.get_global_shape(remained_dim)};
    Contiguous1DMap output_map(output_shape, rank, world_size);
    DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MPI> output ( *vec.copy_comm(), output_map);

    //DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI> output(*vec.copy_comm(), *vec.copy_map());
    
    int m = mat.map.get_local_shape(remained_dim);
    int n = mat.map.get_local_shape(contract_dim);
    std::cout << "m : " << m << " , n : " << n << "vec.map.get_local_shape(0) = " << vec.map.get_local_shape(0) << std::endl;
    double* buffer1;
    double* buffer2;
    if ( trans==TRANSTYPE::T && n == vec.map.get_local_shape(0) ){
        std::cout << "CASE1: multiply and allreduce" << std::endl;
        buffer1 = malloc<double>(m);
        buffer2 = malloc<double>(m);
        
        gemv<double, DEVICETYPE::MPI>(ORDERTYPE::ROW, trans, n,m, 1.0, mat.data, m, vec.data, 1, 0.0, buffer1, 1); 
        
        std::cout << "buffer1 : ";
        for(int i=0;i<m;i++){std::cout << buffer1[i] << ' ';}
        std::cout << std::endl;
        
        output.comm.allreduce(buffer1, m, buffer2, OPTYPE::SUM);
        
        std::cout << "buffer2 : ";
        for(int i=0;i<m;i++){std::cout << buffer2[i] << ' ';}
        std::cout << std::endl;
        
        Gather<Contiguous1DMap<1>>::gather_from_all(buffer2, output);
        free<>(buffer1);
        free<>(buffer2);
    }
    else if ( trans==TRANSTYPE::N && n == vec.map.get_global_shape(0) ){
        std::cout << "CASE2: broadcast vector" << std::endl;
        buffer1 = malloc<double>(vec.map.get_global_shape(0));

        auto all_local_shape = vec.map.get_all_local_shape();
        std::cout << "all_local_shape[i][0] : ";
        size_t recv_counts[world_size];
        for (size_t i=0; i<world_size; i++){
            recv_counts[i] = all_local_shape[i][0];
            std::cout << all_local_shape[i][0] << ' ';
        }
        std::cout << std::endl;

        output.comm.allgatherv(vec.data, all_local_shape[rank][0], buffer1, recv_counts );

        gemv<double, DEVICETYPE::MPI> (ORDERTYPE::ROW, trans, m, n, 1.0, mat.data, n, buffer1, 1, 0.0, output.data, 1);

    }
    else{
        std::cout << "???" <<std::endl;
        //exit(-1);
    }
    return output;
}

template <typename DATATYPE>
DenseTensor<2,DATATYPE,Contiguous1DMap<2>, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, DEVICETYPE::MPI>& mat1,
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, DEVICETYPE::MPI>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N,
    TRANSTYPE trans2=TRANSTYPE::N)
{
    auto world_size = mat1.comm.world_size;
    auto rank = mat1.comm.rank;

    assert (world_size == mat2.comm.world_size);
    assert (rank == mat2.comm.rank);

    size_t contract_dim = (trans1==TRANSTYPE::N)? 1:0;

    assert (  mat1.map.get_global_shape(contract_dim) == mat2.map.get_global_shape(0) );

    std::array<size_t, 2> output_shape = { mat1.map.get_global_size(1-contract_dim)   , mat2.map.get_global_size(1) };
    std::array<bool, 2> is_parallel = {};
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MPI> output(*mat2.copy_comm(), 
                                                                       Contiguous1DMap<2>( output_shape, rank, world_size) );
/*    
    int m = mat1.map.get_global_shape(0);
    int n = mat1.map.get_global_shape(1);

    if (mat2.map.get_split_dim()==0){
        assert (mat2.map.get_global_shape(1)==mat2.map.get_local_shape(1) );
        for (size_t i =0; i<mat2.map.get_global_shape(1); i++){
            TensorOp::matmul<>();
        }
    }
    else{
        assert(false); //not implemented yet.
    }
*/
    return output;
}
////spmv
//template <>
//DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI> SE::TensorOp::matmul<double, Contiguous1DMap<2>, Contiguous1DMap<1>, DEVICETYPE::MPI>(
//    const SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MPI>& mat,
//    const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI>& vec,
//    TRANSTYPE trans)
//{
//
//
//};


/*
template <typename MAPTYPE1, typename MAPTYPE2>
Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* spmv(Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE1>* a, 
                                                         Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* v,
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
                //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size, 0);
                Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(a->comm, return_size, return_data);
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
            //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size);
            Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(v->comm, return_size, return_data);
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
                
                //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size, 0);
                Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(a->comm, return_size, return_data, true, 0);
                
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
            //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size);
            Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(v->comm, return_size, return_data);
            return return_mat;
        }
    }

}   

template <typename MAPTYPE>
Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* spmv(Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* v,
                                                        SE_transpose transa){
    return matmul(a, v, transa, SE_transpose::NoTrans);
}   

template <typename MAPTYPE1, typename MAPTYPE2>
Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* spmv(Tensor<STORETYPE::COO, double, 2, SEMpi, MAPTYPE1>* a, 
                                                        Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>* v,
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
                
                //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(a->comm, return_size, return_data, true, 0);
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
            
            //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(a->comm, return_size, return_data);
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
                //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(a->comm, return_size, return_data, true, 0);
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
            //MAPTYPE2* return_map = new MAPTYPE2(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 1, SEMpi, MAPTYPE2>(v->comm, return_size, return_data);
            return return_mat;
        }
    }
}   

template <typename MAPTYPE>
Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* spmv(Tensor<STORETYPE::COO, double, 2, SEMpi, MAPTYPE>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* v,
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

                //MAPTYPE* return_map = new MAPTYPE(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>(a->comm, return_size, return_data, true, 0);
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

            //MAPTYPE* return_map = new MAPTYPE(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>(a->comm, return_size, return_data);
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
                //MAPTYPE* return_map = new MAPTYPE(return_size, a->comm->world_size, 0);
                auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>(a->comm, return_size, return_data, true, 0);
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
            //MAPTYPE* return_map = new MAPTYPE(return_size, a->comm->world_size);
            auto return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>(a->comm, return_size, return_data);
            return return_mat;
        }
    }
}


//matmul
//alpha * A * B + beta * C, A : m by k, B : k by n, C : m by n
template <typename MAPTYPE>
Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* matmul(Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* a,
                                                          Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* b,
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
        //MAPTYPE* return_map = new MAPTYPE(return_size, a->comm->world_size);
        Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>* return_mat = new Tensor<STORETYPE::Dense, double, 2, SEMpi, MAPTYPE>(a->comm, return_size, return_data);
        return return_mat;
    }
    else{
        std::cout << "to be implemented" << std::endl;
        exit(-1);
    }
}
*/

/*
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
*/


}
