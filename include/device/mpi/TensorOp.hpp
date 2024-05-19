#pragma once
//#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MPIComm.hpp"
#include "../../Gather.hpp"

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))
const int i_zero = 0, i_one = 1, i_negone = -1;

int ictxt;
int info;

namespace SE{
// function declaration is in device/TensorOp.hpp

////dense mv 
//template <>
//DenseTensor<1,double,BlockCyclingMap<1>, DEVICETYPE::MPI> TensorOp::matmul(
//    const DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat,
//    const DenseTensor<1, double, BlockCyclingMap<1>, DEVICETYPE::MPI>& vec,
//    TRANSTYPE trans)
//{
//
//
//    auto world_size = mat.comm.get_world_size();
//    auto rank = mat.comm.get_rank();
//
//    assert (world_size == vec.comm.get_world_size());
//    assert (rank == vec.comm.get_rank());
//
//	char trans_=transtype_to_char(trans);
//	int m = matrix.map.get_global_shape(0);
//	int n = matrix.map.get_global_shape(1);
//	const double one = 1.0;
//	std::array<int, 2> origin = {0,0};
//	auto init_index = matrix.map.local_to_global(origin);
//    int ia = init_index[0];
//    int ja = init_index[1];
//
//	MDESC desca, descx, descy;
//	// block size 
//	int nb = vector.map.get_global_shape(0) / vector.comm.get_world_size();
//
//    descinit_( descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
//	pdgemv(&trans_, &m, &n, &one, mat.data, &ia, &ja, desca  );
//
//
//
//
//
//
//
//
//
//
//
//    assert (  mat.map.get_global_shape(contract_dim) == vec.map.get_global_shape(0) );
//    std::cout << "=============" << std::endl;
//    std::cout << "contract dim  = " << contract_dim << std::endl;
//    std::cout << "mat.global_shape(0) = " << mat.map.get_global_shape(0) << std::endl;
//    std::cout << "mat.global_shape(1) = " << mat.map.get_global_shape(1) << std::endl;
//    std::cout << "vec.global_shape(0) = " << vec.map.get_global_shape(0) << std::endl;
//    std::cout << "=============" << std::endl;
//    
//    std::array<int, 1> output_shape = {mat.map.get_global_shape(remained_dim)};
//    BlockCyclingMap output_map(output_shape, rank, world_size);
//    DenseTensor<1,double,BlockCyclingMap<1>, DEVICETYPE::MPI> output ( *vec.copy_comm(), output_map);
//
//    //DenseTensor<1, double, BlockCyclingMap<1>, DEVICETYPE::MPI> output(*vec.copy_comm(), *vec.copy_map());
//    
//    int m = mat.map.get_local_shape(remained_dim);
//    int n = mat.map.get_local_shape(contract_dim);
//    std::cout << "m : " << m << " , n : " << n << "vec.map.get_local_shape(0) = " << vec.map.get_local_shape(0) << std::endl;
//    double* buffer1;
//    double* buffer2;
//    if ( trans==TRANSTYPE::T && n == vec.map.get_local_shape(0) ){
//        std::cout << "CASE1: multiply and allreduce" << std::endl;
//        buffer1 = malloc<double>(m);
//        buffer2 = malloc<double>(m);
//        
//        gemv<double, DEVICETYPE::MPI>(ORDERTYPE::ROW, trans, n,m, 1.0, mat.data, m, vec.data, 1, 0.0, buffer1, 1); 
//        
//        std::cout << "buffer1 : ";
//        for(int i=0;i<m;i++){std::cout << buffer1[i] << ' ';}
//        std::cout << std::endl;
//        
//        output.comm.allreduce(buffer1, m, buffer2, OPTYPE::SUM);
//        
//        std::cout << "buffer2 : ";
//        for(int i=0;i<m;i++){std::cout << buffer2[i] << ' ';}
//        std::cout << std::endl;
//        
//        Gather<BlockCyclingMap<1>>::gather_from_all(buffer2, output);
//        free<>(buffer1);
//        free<>(buffer2);
//    }
//    else if ( trans==TRANSTYPE::N && n == vec.map.get_global_shape(0) ){
//        std::cout << "CASE2: broadcast vector" << std::endl;
//        buffer1 = malloc<double>(vec.map.get_global_shape(0));
//
//        auto all_local_shape = vec.map.get_all_local_shape();
//        std::cout << "all_local_shape[i][0] : ";
//        int recv_counts[world_size];
//        for (int i=0; i<world_size; i++){
//            recv_counts[i] = all_local_shape[i][0];
//            std::cout << all_local_shape[i][0] << ' ';
//        }
//        std::cout << std::endl;
//
//        output.comm.allgatherv(vec.data, all_local_shape[rank][0], buffer1, recv_counts );
//
//        gemv<double, DEVICETYPE::MPI> (ORDERTYPE::ROW, trans, m, n, 1.0, mat.data, n, buffer1, 1, 0.0, output.data, 1);
//
//    }
//    else{
//        std::cout << "???" <<std::endl;
//        exit(-1);
//    }
//    return output;
//}

template <>
DenseTensor<2,double,BlockCyclingMap<2>, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat1,
    const DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
	const double zero = 0.0;
	const double one = 1.0;

	int desc1[9];
	int desc2[9];
	int desc3[9];

    int m = mat1.map.get_global_shape(0);
    int k = mat1.map.get_global_shape(1);
    if(trans1 != TRANSTYPE::N){
        m= mat1.map.get_global_shape(1);
        k= mat1.map.get_global_shape(0);
    }
    int k2 = mat2.map.get_global_shape(0);
    int n = mat2.map.get_global_shape(1);
    if(trans2 != TRANSTYPE::N){
        k2 = mat2.map.get_global_shape(1);
        n = mat2.map.get_global_shape(0);
    }
    assert(k==k2);

	auto nprow = mat1.map.get_nprow();
	assert (nprow==mat2.map.get_nprow());
	auto block_size = mat1.map.get_block_size();
	assert (block_size == mat2.map.get_block_size());

	int info;
	int row1, col1, row2, col2;
	row1= mat1.map.get_global_shape(0);
	col1= mat1.map.get_global_shape(1);
	row2= mat2.map.get_global_shape(0);
	col2= mat2.map.get_global_shape(1);

	auto row3= m;
	auto col3= n;
	std::array<int,2> new_global_shape = {row3, col3};

	Comm<DEVICETYPE::MPI> new_comm(mat1.comm.get_rank(),mat1.comm.get_world_size() );
	BlockCyclingMap<2> new_map(new_global_shape, new_comm.get_rank(), new_comm.get_world_size(), block_size, nprow );

	DenseTensor<2,double,BlockCyclingMap<2>, DEVICETYPE::MPI> mat3( new_comm, new_map);


    int lld1 = MAX( mat1.map.get_local_shape()[0], 1 );
    int lld2 = MAX( mat2.map.get_local_shape()[0], 1 );
    int lld3 = MAX( mat3.map.get_local_shape()[0], 1 );
    descinit( desc1, &row1, &col1, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert (info==0);
    descinit( desc2, &row2, &col2, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert (info==0);
    descinit( desc3, &row3, &col3, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld3, &info );
	assert (info==0);

	auto trans1_ = transtype_to_char(trans1);
	auto trans2_ = transtype_to_char(trans2);
    pdgemm( &trans1_, 
			&trans2_, 
	        &m, &n, &k, &one, 
			mat1.data, &i_one, &i_one, desc1, 
			mat2.data, &i_one, &i_one, desc2,
            &zero, 
			mat3.data, &i_one, &i_one, desc3 );
	return mat3;
}
////spmv
//template <>
//DenseTensor<1, double, BlockCyclingMap<1>, DEVICETYPE::MPI> SE::TensorOp::matmul<double, BlockCyclingMap<2>, BlockCyclingMap<1>, DEVICETYPE::MPI>(
//    const SparseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat,
//    const DenseTensor<1, double, BlockCyclingMap<1>, DEVICETYPE::MPI>& vec,
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
    int m = a->shape[0];
    int k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<int,1> return_size = {m};

    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k);
        int* recvcounts = v->map.get_partition_size_array();
        v->comm->allgatherv(v->data, recvcounts[v->comm->rank], vector, recvcounts);
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
    int m = a->shape[0];
    int k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<int,1> return_size = {m};


    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k);
        int* recvcounts = v->map.get_partition_size_array();
        v->comm->allgatherv(v->data, recvcounts[v->comm->rank], vector, recvcounts);
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
    int m = a->shape[0];
    int k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    int number_of_vec = v->shape[1];
    std::array<int,2> return_size = {m, number_of_vec};

    if(v->map.is_sliced){
        double* vector = malloc<double, SEMpi>(k*number_of_vec);
        int* recvcounts = v->map.get_partition_size_array();
        for(int n=0; n<number_of_vec ; n++){
            v->comm->allgatherv(&v->data[n*recvcounts[v->comm->rank]], recvcounts[v->comm->rank], &vector[n*k], recvcounts);
        }
        if(a->map.is_sliced){
            if(transa == SE_transpose::NoTrans && a->map.sliced_dimension == 0){
                //case 1
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
                int rank = a->comm->rank;
                int my_size = a->map.get_my_partition_size(rank);
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
    int m = a->shape[0];
    int k = a->shape[1];
    if(transa != SE_transpose::NoTrans){
        m = a->shape[1]; k = a->shape[0];
    }
    int n = b->shape[1];
    if(transb != SE_transpose::NoTrans){
        n = b->shape[0];
        assert(k == b->shape[1]);    
    }
    else{
        assert(k == b->shape[0]);
    }
    std::array<int,2> return_size = {m,n};
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
void orthonormalize<double, SEMpi>(double* eigvec, int vector_size, int number_of_vectors, std::string method)
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

//X + bY
template <>
DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI> SE::TensorOp::add<double, BlockCyclingMap<2>, DEVICETYPE::MPI>(
            DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat1,
            DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat2, double coeff2){
    assert(mat1.map.get_global_shape()[0] == mat2.map.get_global_shape()[0]);
    assert(mat1.map.get_global_shape()[1] == mat2.map.get_global_shape()[1]);

	const double one = 1.0;
	int desc1[9];
	int desc2[9];

    DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI> return_mat(mat1);

	const int row1= mat1.map.get_global_shape(0);
	const int col1= mat1.map.get_global_shape(1);
	const int row2= mat2.map.get_global_shape(0);
	const int col2= mat2.map.get_global_shape(1);

    const int lld1 = MAX( mat1.map.get_local_shape()[0], 1 );
    const int lld2 = MAX( mat2.map.get_local_shape()[0], 1 );

	auto block_size = mat1.map.get_block_size();
	assert (block_size == mat2.map.get_block_size());

    descinit( desc1, &row1, &col1, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert (info==0);
    descinit( desc2, &row2, &col2, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert (info==0);

    const char trans='N';
	pdgeadd( &trans, &row1, &col1, &coeff2, mat2.data, &i_one, &i_one, desc2, &one, return_mat.data, &i_one, &i_one, desc1 );
    return return_mat;
}


}
