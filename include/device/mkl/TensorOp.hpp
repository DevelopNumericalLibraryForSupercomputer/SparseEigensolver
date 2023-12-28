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
    size_t m = mat.map.get_global_shape(0);
    size_t k = mat.map.get_global_shape(1);
    if(trans != TRANSTYPE::N){
        m= mat.map.get_global_shape(1);
        k= mat.map.get_global_shape(0);
    }
    assert ( k == vec.map.get_global_shape(0) );
    
    std::array<size_t, 1> output_shape = {m};
    Contiguous1DMap output_map(output_shape, 0,1);
    DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MKL> output ( *vec.copy_comm(), output_map);
    //mby k * kby 1
    gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, mat.map.get_global_shape(1), vec.data, 1, 0.0, output.data, 1);
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
    std::array<size_t, 2> output_shape = {m,n};
    Contiguous1DMap output_map (output_shape, 0,1);
    DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> output ( *mat2.copy_comm(), output_map );
    //mby k * kby n
    gemm<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, trans1, trans2, m, n, k, 1.0, mat1.data, mat1.map.get_global_shape(1), mat2.data, mat2.map.get_global_shape(1), 0.0, output.data, n);
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

    assert (num_col==vec.map.get_global_shape(0));
    std::array<size_t, 1> output_shape = {num_row};
    Contiguous1DMap output_map (output_shape, 0,1);

    DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> output(*p_comm, output_map );

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
    
    assert (num_col==mat2.map.get_global_shape(0));
    std::array<size_t, 2> output_shape = {num_row, mat2.map.get_global_shape(1)};
    Contiguous1DMap output_map (output_shape, 0,1);

    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> output(*p_comm, output_map );

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
//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <>
//DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> 
void SE::TensorOp::orthonormalize<double, Contiguous1DMap<2>, DEVICETYPE::MKL>( 
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& mat,  
    std::string method)
{
    auto number_of_vectors = mat.map.get_global_shape(1);
    auto vector_size       = mat.map.get_global_shape(0);
    
    
    
    if(method == "qr"){
        double* eigvec = mat.copy_data();
        DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> output ( *mat.copy_comm(), *mat.copy_map() );
        std::unique_ptr<double[]> tau(new double[number_of_vectors]);
        int info = geqrf<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec, number_of_vectors, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        info = orgqr<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, vector_size, number_of_vectors, eigvec, number_of_vectors, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        memcpy<double, DEVICETYPE::MKL>(mat.data, eigvec, number_of_vectors*vector_size);
        free<DEVICETYPE::MKL>(eigvec);
        //return output;
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        auto submatrix = TensorOp::matmul(mat, mat, TRANSTYPE::T, TRANSTYPE::N);
        std::unique_ptr<double[]> submatrix_eigvals(new double[number_of_vectors]);
        syev<double, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'U', number_of_vectors, submatrix.data, number_of_vectors, submatrix_eigvals.get());

        auto output = TensorOp::matmul(mat, submatrix, TRANSTYPE::N, TRANSTYPE::N);
        //vector should be normalized
        for(size_t i=0; i<number_of_vectors; i++){
            double norm = nrm2<double, DEVICETYPE::MKL>(vector_size, &output.data[i], number_of_vectors);
            assert(norm != 0.0);
            scal<double, DEVICETYPE::MKL>(vector_size, 1.0 / norm, &output.data[i], number_of_vectors);
        }
        memcpy<double, DEVICETYPE::MKL>(mat.data, output.data, number_of_vectors*vector_size);
        //return output;
    }
}
}
