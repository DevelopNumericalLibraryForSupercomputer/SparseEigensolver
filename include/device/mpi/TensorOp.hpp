#pragma once
#include "device/mpi/LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MPIComm.hpp"
//#include "../../Gather.hpp"
#include "../../BlockCyclingMap.hpp"

#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"

#include<utility>
/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))


namespace SE{

namespace TensorOp{
const int i_zero = 0, i_one = 1, i_negone = -1;
//int ictxt;  // ictxt is defined in MPIComm.hpp
int info;
};
// function declaration is in device/TensorOp.hpp

//// spmv
//template <typename DATATYPE, MTYPE mtype1, MTYPE mtype2, DEVICETYPE device>
//DenseTensor<1, DATATYPE, mtype2, device> matmul(
//    const SparseTensor<2, DATATYPE, mtype1, device>& mat,
//    const DenseTensor <1, DATATYPE, mtype2, device>& vec,
//    TRANSTYPE trans=TRANSTYPE::N);

//// spmm
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//DenseTensor<2, DATATYPE, mtype, device> matmul(
//    const SparseTensor<2, DATATYPE, mtype, device>& mat1,
//    const DenseTensor <2, DATATYPE, mtype, device>& mat2,
//    TRANSTYPE trans1=TRANSTYPE::N,
//    TRANSTYPE trans2=TRANSTYPE::N);


////Orthonormalization
////n vectors with size m should be stored in m by n matrix (row-major).
////Each coulumn correponds to the vector should be orthonormalized.
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//void TensorOp::orthonormalize(DenseTensor<2, DATATYPE, mtype, device>& mat, std::string method);
//
////y_i = scale_coeff_i * x_i
//template <typename DATATYPE, MTYPE mtype1, DEVICETYPE device>
//void TensorOp::scale_vectors(DenseTensor<2, DATATYPE, mtype1, device>& mat, DATATYPE* scale_coeff);
//
////norm_i = ||mat_i|| (i=0~norm_size-1)
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//void TensorOp::get_norm_of_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat,
//                         DATATYPE* norm, const int norm_size);
//
////mat1_i = mat2_i (i=0~new-1)
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//void TensorOp::copy_vectors(
//        DenseTensor<2, DATATYPE, mtype, device>& mat1,
//        DenseTensor<2, DATATYPE, mtype, device>& mat2, int new_size);
//
////new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//DenseTensor<2, DATATYPE, mtype, device> TensorOp::append_vectors(
//        DenseTensor<2, DATATYPE, mtype, device>& mat1,
//        DenseTensor<2, DATATYPE, mtype, device>& mat2);
//
// // return eigvec
//template <typename DATATYPE, MTYPE mtype1, DEVICETYPE device>
//DenseTensor<2, DATATYPE, mtype1, device> SE::TensorOp::diagonalize(DenseTensor<2, DATATYPE, mtype1, device>& mat, DATATYPE* eigval);



template <>
DenseTensor<1,double,MTYPE::BlockCycling, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat,
    const DenseTensor<1, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& vec,
    TRANSTYPE trans)
{
	const double zero = 0.0;
	const double one = 1.0;

	int desc1[9];
	int desc2[9];
	int desc3[9];

    int m = mat.ptr_map->get_global_shape(0);
    int n = mat.ptr_map->get_global_shape(1);
    if(trans != TRANSTYPE::N){
        m= mat.ptr_map->get_global_shape(1);
        n= mat.ptr_map->get_global_shape(0);
    }
	assert ( n==vec.ptr_map->get_global_shape(0) );

	auto mat_block_size = mat.ptr_map->get_block_size();
	auto vec_block_size = vec.ptr_map->get_block_size();
	//assert (block_size == vec.ptr_map->get_block_size());

	int info;
	int row, col;
	row= mat.ptr_map->get_global_shape(0);
	col= mat.ptr_map->get_global_shape(1);

	std::array<int,1> new_global_shape = { m };
	
	auto comm_inp = mat.ptr_comm->generate_comm_inp();

	auto map_inp = vec.ptr_map->generate_map_inp();
	map_inp->global_shape = new_global_shape;


	DenseTensor<1,double,MTYPE::BlockCycling, DEVICETYPE::MPI> out( comm_inp->create_comm(), map_inp->create_map());

    int lld1 = MAX( mat.ptr_map->get_local_shape()[0], 1 );
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
			mat.data.get(), &i_one, &i_one, desc1, 
			vec.data.get(), &i_one, &i_one, desc2,
            &zero, 
			out.data.get(), &i_one, &i_one, desc3 );
	return out;
	//return std::move(out);
}

template <>
inline
DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> TensorOp::matmul(
    const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
    const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
	const double zero = 0.0;
	const double one = 1.0;

	int desc1[9];
	int desc2[9];
	int desc3[9];

    int m = mat1.ptr_map->get_global_shape(0);
    int k = mat1.ptr_map->get_global_shape(1);
    if(trans1 != TRANSTYPE::N){
        m= mat1.ptr_map->get_global_shape(1);
        k= mat1.ptr_map->get_global_shape(0);
    }
    int k2 = mat2.ptr_map->get_global_shape(0);
    int n = mat2.ptr_map->get_global_shape(1);
    if(trans2 != TRANSTYPE::N){
        k2 = mat2.ptr_map->get_global_shape(1);
        n = mat2.ptr_map->get_global_shape(0);
    }
    assert(k==k2);

	auto nprow = mat1.ptr_map->get_nprow();
	assert (nprow==mat2.ptr_map->get_nprow());
	auto block_size = mat1.ptr_map->get_block_size();
	assert (block_size == mat2.ptr_map->get_block_size());

	int info;
	int row1, col1, row2, col2;
	row1= mat1.ptr_map->get_global_shape(0);
	col1= mat1.ptr_map->get_global_shape(1);
	row2= mat2.ptr_map->get_global_shape(0);
	col2= mat2.ptr_map->get_global_shape(1);

	auto row3= m;
	auto col3= n;
	std::array<int,2> new_global_shape = {row3, col3};

	auto comm_inp = mat1.ptr_comm->generate_comm_inp();

	auto map_inp = mat1.ptr_map->generate_map_inp();
	map_inp->global_shape = new_global_shape;

	DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> mat3( comm_inp->create_comm(),  map_inp->create_map());


    const int lld1 = MAX( mat1.ptr_map->get_local_shape()[0], 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape()[0], 1 );
    const int lld3 = MAX( mat3.ptr_map->get_local_shape()[0], 1 );
               
			   


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
			mat1.data.get(), &i_one, &i_one, desc1, 
			mat2.data.get(), &i_one, &i_one, desc2,
            &zero, 
			mat3.data.get(), &i_one, &i_one, desc3 );
	return mat3;
	//return std::move(mat3);
}

//X + bY
template <>
void TensorOp::add_<double, MTYPE::BlockCycling, DEVICETYPE::MPI>(
            const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
            const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2, const double coeff2){
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    assert(mat1.ptr_map->get_global_shape()[1] == mat2.ptr_map->get_global_shape()[1]);

	const double one = 1.0;
	int desc1[9];
	int desc2[9];

	const int row1= mat1.ptr_map->get_global_shape(0);
	const int col1= mat1.ptr_map->get_global_shape(1);
	const int row2= mat2.ptr_map->get_global_shape(0);
	const int col2= mat2.ptr_map->get_global_shape(1);

    const int lld1 = MAX( mat1.ptr_map->get_local_shape()[0], 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape()[0], 1 );

	auto block_size = mat1.ptr_map->get_block_size();
	assert (block_size == mat2.ptr_map->get_block_size());

    descinit( desc1, &row1, &col1, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert (info==0);
    descinit( desc2, &row2, &col2, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert (info==0);

    const char trans='N';
	pdgeadd( &trans, &row1, &col1, &coeff2, mat2.data.get(), &i_one, &i_one, desc2, &one, mat1.data.get(), &i_one, &i_one, desc1 );
    return;
    //return std::move(return_mat);
}

//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
template <>
void TensorOp::orthonormalize(DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const std::string method){
    //const double one = 1.0;
    int desc[9];
    const int row= mat.ptr_map->get_global_shape(0);
    const int col= mat.ptr_map->get_global_shape(1);
    auto block_size = mat.ptr_map->get_block_size();
    const int lld = MAX( mat.ptr_map->get_local_shape()[0], 1 );
    descinit( desc, &row, &col, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
    assert (info==0);

	auto m = mat.ptr_map->get_global_shape(0);
	auto n = mat.ptr_map->get_global_shape(1);

	int lwork = -1;
	std::vector<double> work(1,0.0);

	double* tau =   malloc<double,DEVICETYPE::MPI>( MIN(m,n)); //  instead new keyword, malloc<> is required 

	//std::cout <<"work(1) " << m << "\t" << n << "\t" <<lwork <<std::endl;
	pdgeqrf(&m, &n, mat.data.get(), &i_one, &i_one, desc, tau, work.data(), &lwork, &info);
	assert(info==0);
	//std::cout <<"work(2) "<< m << "\t" << n << "\t" <<lwork <<std::endl;
    lwork = (int)work[0];
    work.resize(lwork);
	pdgeqrf(&m, &n, mat.data.get(), &i_one, &i_one, desc, tau, work.data(), &lwork, &info);
	assert(info==0);


	lwork=-1;
    pdorgqr(&m, &n, &n, mat.data.get(), &i_one, &i_one, desc, tau, work.data(), &lwork, &info);
	assert(info==0);

	lwork = (int)work[0];
	work.resize(lwork);
    pdorgqr(&m, &n, &n, mat.data.get(), &i_one, &i_one, desc, tau, work.data(), &lwork, &info); // potential problem
	assert(info==0);

	free<DEVICETYPE::MPI>(tau);

    return;

}

//y_i = scale_coeff_i * x_i
template <>
void TensorOp::scale_vectors_(DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const double* scale_coeff){

	for (int i=0; i< mat.ptr_map->get_local_shape(1); i++ ){
		std::array<int,2> local_arr_idx = {0, i};
		auto global_arr_idx = mat.ptr_map->local_to_global(local_arr_idx);
		auto local_idx = mat.ptr_map->unpack_local_array_index(local_arr_idx);
		scal<double, DEVICETYPE::MPI>(mat.ptr_map->get_local_shape(0), scale_coeff[global_arr_idx[1]], &mat.data[local_idx],1);

	}
	return;
}
template <>
void SE::TensorOp::scale_vectors_<double, MTYPE::BlockCycling, DEVICETYPE::MPI>(
            DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const double scale_factor){
	scal<double, DEVICETYPE::MPI>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(),1);
    return;
}

//norm_i = ||mat_i|| (i=0~norm_size-1)
template <>
void TensorOp::get_norm_of_vectors(const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat,
                         double* norm, const int norm_size, const bool root){
    assert(mat.ptr_map->get_global_shape()[1] >= norm_size);

    const int vec_size = mat.ptr_map->get_local_shape()[0];
	const int num_vec  = mat.ptr_map->get_local_shape()[1]; 
	double* local_sum = malloc<double,DEVICETYPE::MPI>(num_vec);
	std::fill_n(local_sum, num_vec, 0.0);

	for (int i=0; i<num_vec; i++){
		#pragma omp parallel for reduction(+:local_sum[i]) 
		for (int j=0; j<vec_size; j++){
			std::array<int, 2> arr = {j,i};
			auto index = mat.ptr_map->unpack_local_array_index(arr);

			local_sum[i] += mat.data[index]*mat.data[index];
		}
	}
	
	std::fill_n(norm, norm_size, 0.0); //initialize norm as 0
	// C means column-wise sum which means summation along processors having the same processor col id is perform
	dgsum2d(&ictxt, "C", "1-tree", &i_one, &num_vec, local_sum, &i_one, &i_negone, &i_negone);


	for (int i =0; i<num_vec; i++){
		std::array<int, 2> arr_idx ={0,i};
		auto col_ind = mat.ptr_map->local_to_global(arr_idx)[1];
		if (col_ind>=0 and col_ind<norm_size){
			if (root){
				norm[col_ind] = sqrt(local_sum[i]);	
			}
			else{
				norm[col_ind] = local_sum[i];	
			}
		}
	}
	free<DEVICETYPE::MPI>(local_sum);
	dgsum2d(&ictxt, "R", "1-tree", &i_one, &norm_size, norm, &i_one, &i_negone, &i_negone);
    return;
}

//norm_i = ||A*B|| (i=0~norm_size-1)
template <>
void TensorOp::vectorwise_dot(const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2,
                               double* norm, const int norm_size){

	assert (mat1.ptr_map->get_global_shape() == mat2.ptr_map->get_global_shape());
	assert (mat1.ptr_map->get_local_shape() == mat2.ptr_map->get_local_shape());
    assert (mat1.ptr_map->get_global_shape()[1] >= norm_size);

	const int local_size =mat1.ptr_map->get_num_local_elements();
	auto buff = malloc<double, DEVICETYPE::MPI>(local_size);

	// element-wise multiplication
	vdMul(local_size, mat1.data.get(), mat2.data.get(), buff);

	// get norm (almost same to the above function)
    const int vec_size = mat1.ptr_map->get_local_shape()[0];
	const int num_vec  = mat1.ptr_map->get_local_shape()[1]; 
	double* local_sum = malloc<double,DEVICETYPE::MPI>(num_vec);
	std::fill_n(local_sum, num_vec, 0.0);
			// debug
//			std::memcpy(mat1.data.get(), buff, local_size*sizeof(double));
//			double* norm_        = malloc<double,DEVICETYPE::MPI> ( num_vec );
//			TensorOp::get_norm_of_vectors(mat1, norm_, num_vec);
//			std::cout << std::setprecision(6) << "norm of mul: " << norm_[0] << ", " << norm_[1] << ", " <<norm_[2] <<std::endl;
//			exit(-1);


	for (int i=0; i<num_vec; i++){
		#pragma omp parallel for reduction(+:local_sum[i]) 
		for (int j=0; j<vec_size; j++){
			//std::array<int, 2> arr = {j,i};
			const auto index = mat1.ptr_map->unpack_local_array_index({j,i});
			local_sum[i] += buff[index];
		}
	}
	
	std::fill_n(norm, norm_size, 0.0); //initialize norm as 0
	// C means column-wise sum which means summation along processors having the same processor col id is perform
	dgsum2d(&ictxt, "C", "1-tree", &i_one, &num_vec, local_sum, &i_one, &i_negone, &i_negone);

	for (int i =0; i<num_vec; i++){
		std::array<int, 2> arr_idx ={0,i};
		auto col_ind = mat1.ptr_map->local_to_global(arr_idx)[1];
		if (col_ind>=0 and col_ind<norm_size){
			norm[col_ind] = local_sum[i];	
		}
	}
	free<DEVICETYPE::MPI>(local_sum);
	// summing up accross the processors (broadcast because norm array is initialized as 0)
	dgsum2d(&ictxt, "R", "1-tree", &i_one, &norm_size, norm, &i_one, &i_negone, &i_negone);
    return;

}


//mat1_i <- mat2_i (i=0~new-1)
template <>
void TensorOp::copy_vectors(
        DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
        DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2, int new_size){

    assert(mat1.ptr_map->get_global_shape()[1] >= new_size);
    assert(mat2.ptr_map->get_global_shape()[1] >= new_size);
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
	const int row = mat1.ptr_map->get_global_shape(0);
	int desc1[9]; const int col1=mat1.ptr_map->get_global_shape(1);
	int desc2[9]; const int col2=mat2.ptr_map->get_global_shape(1);

    const int lld1 = MAX( mat1.ptr_map->get_local_shape(0), 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape(0), 1 );

	auto block_size1 = mat1.ptr_map->get_block_size();
    descinit( desc1, &row, &col1, &block_size1[0], &block_size1[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert(info==0);


	auto block_size2 = mat2.ptr_map->get_block_size();
    descinit( desc2, &row, &col2, &block_size2[0], &block_size2[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert(info==0);

	pdgemr2d(&row, &new_size, mat2.data.get(), &i_one, &i_one, desc2, mat1.data.get(), &i_one, &i_one, desc1, &ictxt);
    return;
	
}

//new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
template<>
std::unique_ptr<DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI> > TensorOp::append_vectors(
        DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
        DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2){


	// define row and check the equaility of mat1 and mat2 row sizes 
	const int row = mat1.ptr_map->get_global_shape(0);
	assert (row==mat2.ptr_map->get_global_shape(0));


	// define desc and col
	int desc1[9]; const int col1=mat1.ptr_map->get_global_shape(1);
	int desc2[9]; const int col2=mat2.ptr_map->get_global_shape(1);
	int desc3[9]; const int col3=col1+col2;

    // new comm (same to the comm of mat1)
	auto comm_inp = mat1.ptr_comm->generate_comm_inp();

    // new map (col3 = col1+col2)
	auto map_inp = mat1.ptr_map->generate_map_inp();
	map_inp->global_shape = {row, col3};

	auto ptr_mat3 = std::make_unique< DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> >( comm_inp->create_comm(), map_inp->create_map());
	//DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> mat3 ( comm_inp->create_comm(), map_inp->create_map());

	//double* local_array = malloc<double,DEVICETYPE::MPI>(ptr_mat3->ptr_map->get_num_local_elements());
    const int lld1 = MAX( mat1.ptr_map->get_local_shape(0), 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape(0), 1 );
    const int lld3 = MAX( ptr_mat3->ptr_map->get_local_shape(0), 1 );

	auto block_size1 = mat1.ptr_map->get_block_size();
    descinit( desc1, &row, &col1, &block_size1[0], &block_size1[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
	assert(info==0);

	auto block_size2 = mat2.ptr_map->get_block_size();
    descinit( desc2, &row, &col2, &block_size2[0], &block_size2[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
	assert(info==0);

	auto block_size3 = mat1.ptr_map->get_block_size();
    descinit( desc3, &row, &col3, &block_size3[0], &block_size3[1], &i_zero, &i_zero, &ictxt, &lld3, &info );
	assert(info==0);

	const int col_idx = i_one + col1; //scalapack use 1 base 
	// mat1->mat3
	pdgemr2d(&row, &col1, mat1.data.get(), &i_one, &i_one, desc1, ptr_mat3->data.get(), &i_one, &i_one, desc3, &ictxt);
	// mat2->mat3
	pdgemr2d(&row, &col2, mat2.data.get(), &i_one, &i_one, desc2, ptr_mat3->data.get(), &i_one, &col_idx, desc3, &ictxt);

	//memcpy<double,DEVICETYPE::MPI>(ptr_mat3->data.get(), local_array, ptr_mat3->ptr_map->get_num_local_elements());
	//free<DEVICETYPE::MPI>(local_array);
	return ptr_mat3;
	//return std::move(mat3);
}

// // return eigvec
template<>
DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI> TensorOp::diagonalize(DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, double* eigval){
	
    assert(mat.ptr_map->get_global_shape()[0] == mat.ptr_map->get_global_shape()[1]);

	// variables
    auto block_size = mat.ptr_map->get_block_size();
	auto global_shape = mat.ptr_map->get_global_shape();
	auto local_shape  = mat.ptr_map->get_local_shape();
    const int lld = MAX(local_shape[0] , 1 );
    //const int lld = MAX( global_shape[0], 1 );
	const int N = global_shape[0]; 
	int desc1[9]; 
	int desc2[9]; 
	
	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();

	// define new matrix for containing eigvec
    DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI> eigvec(mat);

	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();
	// desc1 for mat desc2 for eigvec	
    descinit( desc1, &global_shape[0], &global_shape[1], &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
    descinit( desc2, &global_shape[0], &global_shape[1], &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();

    int lwork = -1, liwork = -1;
    double work_query;
    int iwork_query;
	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();

	// Workspace query for pdsyevd
    pdsyevd("V", "U", &N, mat.data.get(), &i_one, &i_one, desc1, eigval, eigvec.data.get(), &i_one, &i_one, desc2, &work_query, &lwork, &iwork_query, &liwork, &info);
    lwork = (int)work_query;
    liwork = iwork_query;
    std::vector<double> work(lwork);
    std::vector<int> iwork(liwork);
	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();

    // Compute eigenvalues and eigenvectors
    pdsyevd("V", "U", &N, mat.data.get(), &i_one, &i_one, desc1, eigval, eigvec.data.get(), &i_one, &i_one, desc2, work.data(), &lwork, iwork.data(), &liwork, &info);
    assert(info == 0);
	mat.ptr_comm->barrier();
	mat.ptr_comm->barrier();



    return eigvec ;
	
}

}
