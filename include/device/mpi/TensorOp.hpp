#pragma once
//#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MPIComm.hpp"
//#include "../../Gather.hpp"
#include "../../BlockCyclingMap.hpp"

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))
const int i_zero = 0, i_one = 1, i_negone = -1;

int ictxt;
int info;

namespace SE{
// function declaration is in device/TensorOp.hpp

//// spmv
//template <typename DATATYPE, typename MAPTYPE1, typename MAPTYPE2, DEVICETYPE device>
//DenseTensor<1, DATATYPE, MAPTYPE2, device> matmul(
//    const SparseTensor<2, DATATYPE, MAPTYPE1, device>& mat,
//    const DenseTensor <1, DATATYPE, MAPTYPE2, device>& vec,
//    TRANSTYPE trans=TRANSTYPE::N);

//// spmm
//template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
//DenseTensor<2, DATATYPE, MAPTYPE, device> matmul(
//    const SparseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
//    const DenseTensor <2, DATATYPE, MAPTYPE, device>& mat2,
//    TRANSTYPE trans1=TRANSTYPE::N,
//    TRANSTYPE trans2=TRANSTYPE::N);


//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void TensorOp::orthonormalize(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat, std::string method);

//y_i = scale_coeff_i * x_i
template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
void TensorOp::scale_vectors(DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat, DATATYPE* scale_coeff);

//norm_i = ||mat_i|| (i=0~norm_size-1)
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void TensorOp::get_norm_of_vectors(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat,
                         DATATYPE* norm, int norm_size);

//mat1_i = mat2_i (i=0~new-1)
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void TensorOp::copy_vectors(
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2, int new_size);

//new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> TensorOp::append_vectors(
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2);

 // return eigvec
template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE1, device> SE::TensorOp::diagonalize(DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat, DATATYPE* eigval);



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
DenseTensor<1,double,BlockCyclingMap<1>, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat,
    const DenseTensor<1, double, BlockCyclingMap<1>, DEVICETYPE::MPI>& vec,
    TRANSTYPE trans)
{
	const double zero = 0.0;
	const double one = 1.0;

	int desc1[9];
	int desc2[9];
	int desc3[9];

    int m = mat.map.get_global_shape(0);
    int n = mat.map.get_global_shape(1);
    if(trans != TRANSTYPE::N){
        m= mat.map.get_global_shape(1);
        n= mat.map.get_global_shape(0);
    }
	assert ( n==vec.map.get_global_shape(0) );

	auto mat_block_size = mat.map.get_block_size();
	auto vec_block_size = vec.map.get_block_size();
	//assert (block_size == vec.map.get_block_size());

	int info;
	int row, col;
	row= mat.map.get_global_shape(0);
	col= mat.map.get_global_shape(1);

	std::array<int,1> new_global_shape = { m };
	
	Comm<DEVICETYPE::MPI> new_comm(mat.comm.get_rank(),mat.comm.get_world_size() );
	BlockCyclingMap<1> new_map(new_global_shape, new_comm.get_rank(), new_comm.get_world_size(), vec_block_size, vec.map.get_nprow() );

	DenseTensor<1,double,BlockCyclingMap<1>, DEVICETYPE::MPI> out( new_comm, new_map);

    int lld1 = MAX( mat.map.get_local_shape()[0], 1 );
    int lld2 = 1;
    int lld3 = 1;
    descinit( desc1, &row, &col, &mat_block_size[0], &mat_block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert (info==0);
    descinit( desc2, &n, &i_one, &vec_block_size[0], &vec_block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert (info==0);
    descinit( desc3, &n, &i_one, &vec_block_size[0], &vec_block_size[1], &i_zero, &i_zero, &ictxt, &lld3, &info );
	assert (info==0);

	auto trans1 = transtype_to_char(trans);
	const char trans2 = 'N';
    pdgemm( &trans1, &trans2,
	        &m, &i_one, &n, &one, 
			mat.data, &i_one, &i_one, desc1, 
			vec.data, &i_one, &i_one, desc2,
            &zero, 
			out.data, &i_one, &i_one, desc3 );
	return out;
}

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

//X + bY
template <>
DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI> TensorOp::add<double, BlockCyclingMap<2>, DEVICETYPE::MPI>(
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

//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void TensorOp::orthonormalize(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat, std::string method){

}

//y_i = scale_coeff_i * x_i
void TensorOp::scale_vectors(DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat, double* scale_coeff){

	const double one = 1.0;
	int desc[9];
	const int row= mat.map.get_global_shape(0);
	const int col= mat.map.get_global_shape(1);
	auto block_size = mat.map.get_block_size();
    const int lld = MAX( mat.map.get_local_shape()[0], 1 );

    descinit( desc, &row, &col, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
	assert (info==0);

    const int vec_size = mat.map.get_global_shape()[0];
	const int num_vec  = mat.map.get_global_shape()[1]; 
    for(int index = 0; index < num_vec; index++){
        //scal<double, DEVICETYPE::MKL>(vec_size, scale_coeff[index], &mat.data[index], block_size);
		pdscal ( &vec_size, &scale_coeff[index] , mat.data, &i_one, &index, desc, &num_vec); 
    }
	return;
}

//norm_i = ||mat_i|| (i=0~norm_size-1)
void TensorOp::get_norm_of_vectors(DenseTensor<2, double, BlockCyclingMap<2>, DEVICETYPE::MPI>& mat,
                         double* norm, int norm_size){
    assert(mat.map.get_global_shape()[1] >= norm_size);

    const int vec_size = mat.map.get_local_shape()[0];
	const int num_vec  = mat.map.get_local_shape()[1]; 
	double* local_sum = new double[num_vec]();


	for (int i=0; i<num_vec; i++){
		#pragma omp parallel for reduction(+:local_sum[i]) 
		for (int j=0; j<vec_size; j++){
			std::array<int, 2> arr = {j,i};
			auto index = mat.map.unpack_local_array_index(arr);
			local_sum[i] += mat.data[index]*mat.data[index];
		}
	}
	
	std::fill_n(norm, norm_size, 0.0); //initialize norm as 0

	dgsum2d(&ictxt, "C", "1-tree", &i_one, &num_vec, local_sum, &i_one, &i_negone, &i_negone);


	for (int i =0; i<num_vec; i++){
		std::array<int, 2> arr_idx ={0,i};
		auto col_ind = mat.map.local_to_global(arr_idx)[1];
		if (col_ind>=0 and col_ind<norm_size){
			norm[col_ind] = sqrt(local_sum[i]);	
		}
	}

	dgsum2d(&ictxt, "R", "1-tree", &i_one, &norm_size, norm, &i_one, &i_negone, &i_negone);
    return;
}

//
////mat1_i = mat2_i (i=0~new-1)
//template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
//void copy_vectors(
//        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
//        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2, int new_size);
//
////new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
//template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
//DenseTensor<2, DATATYPE, MAPTYPE, device> append_vectors(
//        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
//        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2);
//
// // return eigvec
//template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
//DenseTensor<2, DATATYPE, MAPTYPE1, device> diagonalize(DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat, DATATYPE* eigval);

}
