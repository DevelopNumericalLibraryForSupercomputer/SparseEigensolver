#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MKLComm.hpp"

#include "mkl_types.h"
#include "mkl_spblas.h"
namespace SE{

// dense mv
template <>
DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MKL> TensorOp::matmul(
//DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MKL> TensorOp::matmul<double, Contiguous1DMap<2>, Contiguous1DMap<1>, DEVICETYPE::MKL>(
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat,
    const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec,
    TRANSTYPE trans)
{
    assert ( mat.map.get_global_shape(1) == vec.map.get_global_shape(0) );
    DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MKL> output ( *vec.copy_comm(), *vec.copy_map() );

    size_t m = mat.map.get_global_shape(0);
    size_t k = mat.map.get_global_shape(1);
    if(trans != TRANSTYPE::N){
        m= mat.map.get_global_shape(1);
        k= mat.map.get_global_shape(0);
    }
    //mby k * kby n
    gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, k, vec.data, 1, 0.0, output.data, 1);
    return output;
}

// dense mm
template <>
DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> TensorOp::matmul(
//DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> TensorOp::matmul<double, Contiguous1DMap<2>, DEVICETYPE::MKL>(
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat1,
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
    assert ( mat1.map.get_global_shape(1) == mat2.map.get_global_shape(0) );
    DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> output ( *mat2.copy_comm(), *mat2.copy_map() );

    size_t m = mat1.map.get_global_shape(0);
    size_t k = mat1.map.get_global_shape(1);
    if(trans1 != TRANSTYPE::N){
        m= mat1.map.get_global_shape(1);
        k= mat1.map.get_global_shape(0);
    }
    size_t k2 = mat2.map.get_global_shape(0);
    size_t n = mat2.map.get_global_shape(1);
    if(trans2 != TRANSTYPE::N){
        k2 = mat2.map.get_global_shape(1);
        n = mat2.map.get_global_shape(0);
    }
    assert(k == k2);

    //mby k * kby n
    gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans1, trans2, m, n, k, 1.0, mat1.data, k, mat2.data, n, 0.0, output.data, n);
    return output;
}

// sparse mv
template <>
DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> SE::TensorOp::matmul<double, Contiguous1DMap<2>, Contiguous1DMap<1>, DEVICETYPE::MKL>(
    const SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat,
    const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec,
    TRANSTYPE trans)
{
    sparse_matrix_t cooA;
    struct matrix_descr descrA;
    sparse_status_t status;

    auto shape = mat.map.get_global_shape();
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

    auto p_comm=vec.copy_comm();
    auto p_map =vec.copy_map();

    assert (mat.map.get_global_shape(1)==p_map->get_global_shape(0));
    DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> output(*p_comm, *p_map );

    status = mkl_sparse_d_create_coo( &cooA,
                                      SPARSE_INDEX_BASE_ZERO,
                                      num_row,    // number of rows
                                      num_col,    // number of cols
                                      nnz,  // number of nonzeros
                                      mat.complete_index,
                                      mat.complete_index+nnz,
                                      mat.complete_value );

    assert (status == SPARSE_STATUS_SUCCESS);

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;

    if(trans == TRANSTYPE::N){
        status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, vec.data, 0.0, output.data);
    }
    else{
        status = mkl_sparse_d_mv( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, vec.data, 0.0, output.data);
    }
    assert (status == SPARSE_STATUS_SUCCESS);

    return output;
}

// sparse mm 
template <>
DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> SE::TensorOp::matmul<double, Contiguous1DMap<2>, DEVICETYPE::MKL>(
    const SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat1,
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
    // the code for the case with trans2 ==TRANSTYPE::Y is not yet developed
    assert (trans2 == TRANSTYPE::N);
    sparse_matrix_t cooA;
    struct matrix_descr descrA;
    sparse_status_t status;

    auto shape = mat1.map.get_global_shape();
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

    auto p_comm=mat2.copy_comm();
    auto p_map =mat2.copy_map();

    assert (mat1.map.get_global_shape(1)==p_map->get_global_shape(0));
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> output(*p_comm, *p_map );

    status = mkl_sparse_d_create_coo( &cooA,
                                      SPARSE_INDEX_BASE_ZERO,
                                      num_row,    // number of rows
                                      num_col,    // number of cols
                                      nnz, // number of nonzeros
                                      mat1.complete_index,
                                      mat1.complete_index+nnz,
                                      mat1.complete_value );

    assert (status == SPARSE_STATUS_SUCCESS);

    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA.diag = SPARSE_DIAG_NON_UNIT;

    if(trans1 == TRANSTYPE::N){
        status = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, SPARSE_LAYOUT_ROW_MAJOR, mat2.data, num_col,num_col, 1.0, output.data, num_col);
    }
    else{
        status = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, SPARSE_LAYOUT_ROW_MAJOR, mat2.data, num_col, num_col, 1.0, output.data, num_col);
    }
    assert (status == SPARSE_STATUS_SUCCESS);

    return output;
}
//orthonormalize
template <>
DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> SE::TensorOp::orthonormalize<double, Contiguous1DMap<2>, DEVICETYPE::MKL>( 
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat,  
    std::string method)
{
    auto number_of_vectors = mat.map.get_global_shape(1);
    auto vector_size       = mat.map.get_global_shape(0);
    double* eigvec         = mat.copy_data();
    
    
    if(method == "qr"){
        DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> output ( *mat.copy_comm(), *mat.copy_map() );
        //double* tau = malloc<double, DEVICETYPE::MKL>(number_of_vectors);
        std::unique_ptr<double[]> tau(new double[number_of_vectors]);
        int info = geqrf<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec, vector_size, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        info = orgqr<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, number_of_vectors, number_of_vectors, eigvec, vector_size, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        //free<double, DEVICETYPE::MKL>(tau);
        //return DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>(mat.copy_comm(), mat.copy_map(), eigvec);
        memcpy<double, DEVICETYPE::MKL>(output.data, eigvec, number_of_vectors*vector_size, COPYTYPE::NONE);
        free<DEVICETYPE::MKL>(eigvec);
        return output;
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        //double* submatrix = malloc<double, DEVICETYPE::MKL>(number_of_vectors*number_of_vectors);
        //gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, TRANSTYPE::T, TRANSTYPE::N, number_of_vectors, number_of_vectors, vector_size, 1.0, eigvec, number_of_vectors, eigvec, vector_size, 0.0, submatrix.get(), number_of_vectors);
        auto submatrix = TensorOp::matmul(mat, mat, TRANSTYPE::T, TRANSTYPE::N);
        std::cout << "submatrix" << std::endl;
        std::cout << submatrix;
        //double* submatrix_eigvals = malloc<double, DEVICETYPE::MKL>(number_of_vectors);
        std::unique_ptr<double[]> submatrix_eigvals(new double[number_of_vectors]);
        syev<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'U', number_of_vectors, submatrix.data, number_of_vectors, submatrix_eigvals.get());
        std::cout << "submatrix" << std::endl;
        std::cout << submatrix;
        return TensorOp::matmul(mat, submatrix, TRANSTYPE::N, TRANSTYPE::N);
        //double* new_eigvec = malloc<double, DEVICETYPE::MKL>(vector_size*number_of_vectors);
        //gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, TRANSTYPE::N, TRANSTYPE::N,vector_size, number_of_vectors, number_of_vectors, 1.0, eigvec, vector_size, submatrix, number_of_vectors, 0.0, new_eigvec, vector_size);
        //free<double, DEVICETYPE::MKL>(new_eigvec);
    }
}
}
