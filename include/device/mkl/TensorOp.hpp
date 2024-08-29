#pragma once
#include "device/mkl/LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MKLComm.hpp"

#include "mkl_types.h"
#include "mkl_spblas.h"
namespace SE{

// dense mv
template <>
std::unqiue_ptr<DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> > TensorOp::matmul(
//DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> TensorOp::matmul<double, MTYPE::Contiguous1D, Map<1,MTYPE::Contiguous1D>, DEVICETYPE::MKL>(
    const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat,
    const DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec,
    TRANSTYPE trans)
{
    
    int m = mat.ptr_map->get_global_shape(0);
    int k = mat.ptr_map->get_global_shape(1);

    int contract_dim = (trans==TRANSTYPE::N)? 1:0;
    int remained_dim = (trans==TRANSTYPE::N)? 0:1;
    /*
    if(trans != TRANSTYPE::N){
        m= mat.ptr_map->get_global_shape(1);
        k= mat.ptr_map->get_global_shape(0);
    }
    */
    assert ( mat.ptr_map->get_global_shape(contract_dim) == vec.ptr_map->get_global_shape(0) );
    std::array<int, 1> output_shape = {mat.ptr_map->get_global_shape(remained_dim)};
	std::unique_ptr<Map<1,MTYPE::Contiguous1D>> ptr_output_map = std::make_unique<Contiguous1DMap<1>> (output_shape, 0,1);
    std::unique_ptr<DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> > output = std::make_unique<DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> > ( vec.copy_comm(), ptr_output_map);
    //mby k * kby 1
    //gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, mat.ptr_map->get_global_shape(1), vec.data, 1, 0.0, output.data, 1);
    gemv<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, m, k, 1.0, mat.data.get(), mat.ptr_map->get_global_shape(1), vec.data.get(), 1, 0.0, output->data.get(), 1);
    return output;
}

// dense mm
template <>
std::unique_ptr<DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> > TensorOp::matmul<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
//DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> TensorOp::matmul<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
    const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,
    const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
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
    
    assert(k == k2);
    std::array<int, 2> output_shape = {m,n};

	std::unique_ptr<Map<2,MTYPE::Contiguous1D>> ptr_output_map = std::make_unique<Contiguous1DMap<2>> (output_shape, 0,1);
    std::unique_ptr<DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> > output =std::make_unique<DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> >( mat2.copy_comm(), ptr_output_map );
    //mby k * kby n
    gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans1, trans2, m, n, k, 1.0, mat1.data.get(), mat1.ptr_map->get_global_shape(1), mat2.data.get(), mat2.ptr_map->get_global_shape(1), 0.0, output->data.get(), n);
    return output;
}

// sparse mv
template <>
std::unique_ptr<DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > SE::TensorOp::matmul<double, MTYPE::Contiguous1D, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
    const SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat,
    const DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec,
    TRANSTYPE trans)
{
    sparse_matrix_t cooA;
    struct matrix_descr descrA;
    sparse_status_t status;

    auto shape = mat.ptr_map->get_global_shape();
    auto nnz = mat.get_num_nonzero();

    int num_row;
    int num_col;
    int* row_indx;
    int* col_indx;
    if(trans == TRANSTYPE::N){
        row_indx = &mat.complete_index[0];
        col_indx = &mat.complete_index[nnz];
        num_row = shape[0];
        num_col = shape[1];
    }
    else{
        row_indx = &mat.complete_index[nnz];
        col_indx = &mat.complete_index[0];
        num_row = shape[1];
        num_col = shape[0];
    }    

    auto ptr_comm=vec.copy_comm();

    assert (num_col==vec.ptr_map->get_global_shape(0));
    //std::array<int, 1> output_shape = {static_cast<unsigned long>(num_row)};
    std::array<int, 1> output_shape = {num_row};
	std::unique_ptr<Map<1,MTYPE::Contiguous1D>> ptr_output_map = std::make_unique<Contiguous1DMap<1>> (output_shape, 0,1);

    std::unique_ptr<DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > output = std::make_unique<DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>>(ptr_comm, ptr_output_map );

    status = mkl_sparse_d_create_coo( &cooA,
                                      SPARSE_INDEX_BASE_ZERO,
                                      num_row,    // number of rows
                                      num_col,    // number of cols
                                      nnz,  // number of nonzeros
                                      mat.complete_index.get(),
                                      mat.complete_index.get()+nnz,
                                      mat.complete_value.get() );

    assert (status == SPARSE_STATUS_SUCCESS);

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;

    if(trans == TRANSTYPE::N){
        status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, vec.data.get(), 0.0, output->data.get());
    }
    else{
        status = mkl_sparse_d_mv( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, vec.data.get(), 0.0, output->data.get());
    }
    assert (status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(cooA);
    assert (status == SPARSE_STATUS_SUCCESS);

    return output;
}

// sparse mm 
template <>
std::unique_ptr< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > SE::TensorOp::matmul<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
    const SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,
    const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
    // the code for the case with trans2 ==TRANSTYPE::Y is not yet developed
    assert (trans2 == TRANSTYPE::N);
    sparse_matrix_t cooA;
    struct matrix_descr descrA;
    sparse_status_t status;

    auto shape = mat1.ptr_map->get_global_shape();
    auto nnz = mat1.get_num_nonzero();

    int num_row;
    int num_col;
    int* row_indx;
    int* col_indx;

    if(trans1 == TRANSTYPE::N){
        row_indx = &mat1.complete_index[0];
        col_indx = &mat1.complete_index[nnz];
        num_row = shape[0];
        num_col = shape[1];
    }
    else{
        row_indx = &mat1.complete_index[nnz];
        col_indx = &mat1.complete_index[0];
        num_row = shape[1];
        num_col = shape[0];
    }    

    auto ptr_comm=mat2.copy_comm();
    
    assert (num_col==mat2.ptr_map->get_global_shape(0));
    //std::array<int, 2> output_shape = {static_cast<unsigned long>(num_row), mat2.ptr_map->get_global_shape(1)};
    std::array<int, 2> output_shape = {num_row, mat2.ptr_map->get_global_shape(1)};
	std::unique_ptr<Map<2,MTYPE::Contiguous1D>> ptr_output_map = std::make_unique<Contiguous1DMap<2>> (output_shape, 0,1);

    std::unique_ptr<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > output = std::make_unique<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (ptr_comm, ptr_output_map );

    status = mkl_sparse_d_create_coo( &cooA,
                                      SPARSE_INDEX_BASE_ZERO,
                                      num_row,    // number of rows
                                      num_col,    // number of cols
                                      nnz, // number of nonzeros
                                      mat1.complete_index.get(),
                                      mat1.complete_index.get()+nnz,
                                      mat1.complete_value.get() );

    assert (status == SPARSE_STATUS_SUCCESS);

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;
//sparse_status_t mkl_sparse_d_mm (const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const sparse_layout_t layout, const double *B, const MKL_INT columns, const MKL_INT ldb, const double beta, double *C, const MKL_INT ldc);

    if(trans1 == TRANSTYPE::N){
        status = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 
		                          1.0, 
								  cooA, 
								  descrA, 
								  SPARSE_LAYOUT_ROW_MAJOR, 
								  mat2.data.get(), 
								  mat2.ptr_map->get_global_shape(1), 
								  mat2.ptr_map->get_global_shape(1), 
								  0.0, 
								  output->data.get(), 
								  output->ptr_map->get_global_shape(1));
    }
    else{
        status = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,     
		                          1.0, 
								  cooA, 
								  descrA, 
								  SPARSE_LAYOUT_ROW_MAJOR, 
								  mat2.data.get(), 
								  mat2.ptr_map->get_global_shape(1), 
								  mat2.ptr_map->get_global_shape(1), 
								  0.0, 
								  output->data.get(), 
								  output->ptr_map->get_global_shape(1));
    }
    assert (status == SPARSE_STATUS_SUCCESS);
    status = mkl_sparse_destroy(cooA);
    assert (status == SPARSE_STATUS_SUCCESS);
    return output;
}


//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <>
//DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> 
void SE::TensorOp::orthonormalize<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>( 
    DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat,  
    const std::string method)
{
    auto number_of_vectors = mat.ptr_map->get_global_shape(1);
    auto vector_size       = mat.ptr_map->get_global_shape(0);
    
    if(method == "qr"){
        auto eigvec = mat.copy_data();
        DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> output ( mat.copy_comm(), mat.copy_map() );
        std::unique_ptr<double[]> tau(new double[number_of_vectors]);
        int info = geqrf<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec.get(), number_of_vectors, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        info = orgqr<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec.get(), number_of_vectors, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        memcpy<double, DEVICETYPE::MKL>(mat.data.get(), eigvec.get(), number_of_vectors*vector_size);
        
    }
    else{
        auto submatrix = TensorOp::matmul(mat, mat, TRANSTYPE::T, TRANSTYPE::N);
        std::unique_ptr<double[]> submatrix_eigvals(new double[number_of_vectors]);
        syev<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'U', number_of_vectors, submatrix.data.get(), number_of_vectors, submatrix_eigvals.get());

        auto output = TensorOp::matmul(mat, submatrix, TRANSTYPE::N, TRANSTYPE::N);
        //vector should be normalized
        for(int i=0; i<number_of_vectors; i++){


//gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, mat.ptr_map->get_global_shape(1), vec.data, 1, 0.0, output.data, 1);
//gemv<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, m, k, 1.0, mat.data.get(), mat.ptr_map->get_global_shape(1), vec.data.get(), 1, 0.0, output.data.get(), 1);

            double norm = nrm2<double, DEVICETYPE::MKL>(vector_size, &output.data[i], number_of_vectors);
            assert(norm != 0.0);
            scal<double, DEVICETYPE::MKL>(vector_size, 1.0 / norm, &output.data[i], number_of_vectors);
        }
        memcpy<double, DEVICETYPE::MKL>(mat.data.get(), output.data.get(), number_of_vectors*vector_size);
        
    }
}

template <>
void SE::TensorOp::scale_vectors_<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
            DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat, const double* scale_coeff){
    if(scale_coeff == nullptr){
        std::cout << "WRONG scale_coeff in TensorOp::scale_vectors!" << std::endl;
        exit(1);
    }
    int vec_size = mat.ptr_map->get_global_shape()[0];
    int num_vec = mat.ptr_map->get_global_shape()[1];
    for(int index = 0; index < num_vec; index++){
        scal<double, DEVICETYPE::MKL>(vec_size, scale_coeff[index], &mat.data[index], num_vec);
    }
    return;
}

template <>
void SE::TensorOp::scale_vectors_<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
            DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat, const double scale_factor){
    scal<double, DEVICETYPE::MKL>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(), 1);
    return;
}


//X + bY
template <>
void SE::TensorOp::add_<double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(
            const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,
            const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2, const double coeff2){
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    assert(mat1.ptr_map->get_global_shape()[1] == mat2.ptr_map->get_global_shape()[1]);
    axpy<double, DEVICETYPE::MKL>(mat1.ptr_map->get_global_shape()[0]*mat1.ptr_map->get_global_shape()[1],coeff2,mat2.data.get(),1,mat1.data.get(),1);
    return;
}

template <>
void SE::TensorOp::get_norm_of_vectors(const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat, double* norm, const int norm_size, const bool root /*=true*/){
    assert(mat.ptr_map->get_global_shape()[1] >= norm_size);
    for(int i=0;i<norm_size;i++){
        norm[i] = nrm2<double, DEVICETYPE::MKL>(mat.ptr_map->get_global_shape()[0], &mat.data[i], mat.ptr_map->get_global_shape()[1]);
		if (root==false){
			norm[i] = norm[i]*norm[i];	
		}
    }
    return;
}

//norm_i = ||A*B|| (i=0~norm_size-1)
template <>
void SE::TensorOp::vectorwise_dot(const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2,
                               double* norm, const int norm_size){

	assert (mat1.ptr_map->get_global_shape() == mat2.ptr_map->get_global_shape());
	assert (mat1.ptr_map->get_local_shape() == mat2.ptr_map->get_local_shape());
    assert (mat1.ptr_map->get_global_shape()[1] >= norm_size);


    const int vec_size = mat1.ptr_map->get_local_shape()[0];
	const int num_vec  = mat1.ptr_map->get_local_shape()[1]; 

	const int local_size =mat1.ptr_map->get_num_local_elements();
	double* buff = malloc<double,DEVICETYPE::MKL>(local_size);
	vdMul(local_size, mat1.data.get(), mat2.data.get(), buff);

	std::fill_n(norm, num_vec, 0.0);
    
    for(int i=0;i<norm_size;i++){
		#pragma omp parallel for reduction(+:norm[i]) 
		for (int j=0; j<vec_size; j++){
			//std::array<int, 2> arr = {j,i};
			const auto index = mat1.ptr_map->unpack_local_array_index({j,i});
			norm[i] += buff[index];
		}

    }
	free<DEVICETYPE::MKL>(buff);
	return;
}

template <>
void SE::TensorOp::copy_vectors(
        DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,
        DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2, int new_size){
    assert(mat1.ptr_map->get_global_shape()[1] >= new_size);
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    int vec_size = mat1.ptr_map->get_global_shape()[0];
    for(int i=0;i<new_size;i++){
        copy<double, DEVICETYPE::MKL>(vec_size, &mat2.data[i], mat2.ptr_map->get_global_shape()[1], &mat1.data[i], mat1.ptr_map->get_global_shape()[1]);
    }
    return;
}

template <>
std::unique_ptr<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > SE::TensorOp::append_vectors(
        DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat1,
        DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat2){
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);

    std::array<int, 2> new_shape = {mat1.ptr_map->get_global_shape()[0], mat1.ptr_map->get_global_shape()[1] + mat2.ptr_map->get_global_shape()[1]};
    std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_new_map = std::make_unique<Contiguous1DMap<2> >(new_shape, mat1.ptr_comm->get_rank(), mat1.ptr_comm->get_world_size());
	auto mat = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (mat1.copy_comm(), ptr_new_map);
    //auto ptr_mat =std::make_unique<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (mat1.copy_comm(), ptr_new_map);
    for(int i=0;i<mat1.ptr_map->get_global_shape()[1];i++){
        copy<double, DEVICETYPE::MKL>(new_shape[0], &mat1.data[i], mat1.ptr_map->get_global_shape()[1], &(mat->data[i]), new_shape[1]);
    }
    for(int i=0;i<mat2.ptr_map->get_global_shape()[1];i++){
        copy<double, DEVICETYPE::MKL>(new_shape[0], &mat2.data[i], mat2.ptr_map->get_global_shape()[1], &(mat->data[i+mat1.ptr_map->get_global_shape()[1]]), new_shape[1]);
    }
    return mat;
}


 // return eigvec
template <>
std::unique_ptr<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > SE::TensorOp::diagonalize(
        DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& mat, double* eigval){
    assert(mat.ptr_map->get_global_shape()[0] == mat.ptr_map->get_global_shape()[1]);
    int block_size = mat.ptr_map->get_global_shape()[0];
    auto eigvec = std::make_unique<DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (mat);
    int info = syev<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'U', block_size, eigvec->data.get(), block_size, eigval);
    if(info !=0){
        std::cout << "subspace_diagonalization error!" << std::endl;
        exit(-1);
    }
    return eigvec;
}





}
