#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "MKLComm.hpp"

#include "mkl_types.h"
#include "mkl_spblas.h"
namespace SE{
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
        status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, vec.data, 1.0, output.data);
    }
    else{
        status = mkl_sparse_d_mv( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, vec.data, 1.0, output.data);
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
    std::cout << "Start" <<std::endl;
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

    std::cout << "000000000" <<std::endl;
    auto p_comm=mat2.copy_comm();
    std::cout << "333333333" <<std::endl;
    auto p_map =mat2.copy_map();

    assert (mat1.map.get_global_shape(1)==p_map->get_global_shape(0));
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> output(*p_comm, *p_map );

    std::cout << "11111111" <<std::endl;
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
        std::cout << "2222222222" <<std::endl;
        status = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, SPARSE_LAYOUT_ROW_MAJOR, mat2.data, num_col,num_col, 1.0, output.data, num_col);
    }
    else{
        status = mkl_sparse_d_mm( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, SPARSE_LAYOUT_ROW_MAJOR, mat2.data, num_col, num_col, 1.0, output.data, num_col);
    }
    assert (status == SPARSE_STATUS_SUCCESS);

    return output;
}
////QR
//template <>
//void SE::TensorOp::orthonormalize<double, Contiguous1DMap<2>, DEVICETYPE::MKL>( 
//    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat,  
//    std::string method)
//{
//    auto number_of_vectors = mat.map.get_global_shape(1);
//    auto vector_size       = mat.map.get_global_shape(0);
//
//    if(method == "qr"){
//        double* tau = malloc<double, DEVICETYPE::MKL>(number_of_vectors);
//        int info = geqrf<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec, vector_size, tau);
//        if(info != 0){
//            std::cout << "QR decomposition failed!" << std::endl;
//            exit(1);
//        }
//        info = orgqr<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
//        if(info != 0){
//            std::cout << "QR decomposition failed!" << std::endl;
//            exit(1);
//        }
//        free<double, DEVICETYPE::MKL>(tau);
//    }
//    else{
//        std::cout << "default orthonormalization" << std::endl;
//        double* submatrix = malloc<double, DEVICETYPE::MKL>(number_of_vectors*number_of_vectors);
//        double* submatrix_eigvals = malloc<double, DEVICETYPE::MKL>(number_of_vectors);
//        gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, TRANSTYPE::T, TRANSTYPE::N, number_of_vectors, number_of_vectors, vector_size, 1.0, eigvec, number_of_vectors, eigvec, vector_size, 0.0, submatrix, number_of_vectors);
//        syev<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'U', number_of_vectors, submatrix, number_of_vectors, submatrix_eigvals);
//        double* new_eigvec = malloc<double, DEVICETYPE::MKL>(vector_size*number_of_vectors);
//        gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, TRANSTYPE::N, TRANSTYPE::N,vector_size, number_of_vectors, number_of_vectors, 1.0, eigvec, vector_size, submatrix, number_of_vectors, 0.0, new_eigvec, vector_size);
//        memcpy<double, DEVICETYPE::MKL>(eigvec, new_eigvec, vector_size*number_of_vectors);
//        free<double, DEVICETYPE::MKL>(submatrix);
//        free<double, DEVICETYPE::MKL>(submatrix_eigvals);
//        free<double, DEVICETYPE::MKL>(new_eigvec);
//    }
//}
}
