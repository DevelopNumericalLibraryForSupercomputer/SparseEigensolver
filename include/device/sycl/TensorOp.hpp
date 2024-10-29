#pragma once
#include "device/sycl/LinearOp.hpp"
#include "device/TensorOp.hpp"
#include "SYCLComm.hpp"
#include "Contiguous1DColMap.hpp"
//#include "mkl_types.h"
//#include "mkl_spblas.h"
namespace SE{

// dense mv
template <>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<1,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul(
//DenseTensor<1,DATATYPE,MTYPE::, DEVICETYPE::SYCL> TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul<DATATYPE, MTYPE::Contiguous1DCol, Map<1,MTYPE::Contiguous1DCol>, DEVICETYPE::SYCL>(
    const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat,
    const DenseTensor<1, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& vec,
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
    assert (trans==TRANSTYPE::N);
    assert ( mat.ptr_map->get_global_shape(contract_dim) == vec.ptr_map->get_global_shape(0) );
    std::array<int, 1> output_shape = {mat.ptr_map->get_global_shape(remained_dim)};
    std::unique_ptr<Map<1,MTYPE::Contiguous1DCol>> ptr_output_map = std::make_unique<Contiguous1DColMap<1>> (output_shape, 0,1);
    std::unique_ptr<DenseTensor<1,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > output = std::make_unique<DenseTensor<1,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > ( vec.copy_comm(), ptr_output_map);
    //mby k * kby 1
    //gemm<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, mat.ptr_map->get_global_shape(1), vec.data, 1, 0.0, output.data, 1);
    gemv<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, trans, m, k, 1.0, mat.data.get(), m, vec.data.get(), 1, 0.0, output->data.get(), 1);
    return output;
}

// dense mm
template <>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul(
//DenseTensor<2,DATATYPE,MTYPE::, DEVICETYPE::SYCL> TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul<DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>(
    const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat1,
    const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat2,
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

    std::unique_ptr<Map<2,MTYPE::Contiguous1DCol>> ptr_output_map = std::make_unique<Contiguous1DColMap<2>> (output_shape, 0,1);
    std::unique_ptr<DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > output =std::make_unique<DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> >( mat2.copy_comm(), ptr_output_map );
    //mby k * kby n
    
    int lda = (trans1 == TRANSTYPE::N ? m:k );
    int ldb = (trans2 == TRANSTYPE::N ? k2:n );
    int ldc = m;
    gemm<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, trans1, trans2, m, n, k, 1.0, mat1.data.get(), lda, mat2.data.get(), ldb, 0.0, output->data.get(), ldc);
    return output;
}

////////////////////////////////////////////////////// Sparse type helper functions 
//template<typename DATATYPE>
//sparse_status_t create_coo(sparse_matrix_t *A, 
//                           const sparse_index_base_t indexing, 
//                           const int rows, 
//                           const int cols, 
//                           const int nnz, 
//                           int *row_indx, 
//                           int * col_indx, 
//                           DATATYPE *values);
//
//template<>
//sparse_status_t create_coo<double>(sparse_matrix_t *A, 
//                                   const sparse_index_base_t indexing, 
//                                   const int rows, 
//                                   const int cols, 
//                                   const int nnz, 
//                                   int *row_indx, 
//                                   int * col_indx, 
//                                   double *values){
//    return mkl_sparse_d_create_coo(A,indexing,rows,cols,nnz,row_indx,col_indx,values);    
//}
//
//template<>
//sparse_status_t create_coo<std::complex<double> >(sparse_matrix_t *A, 
//                                                  const sparse_index_base_t indexing, 
//                                                  const int rows, 
//                                                  const int cols, 
//                                                  const int nnz, 
//                                                  int *row_indx, 
//                                                  int * col_indx, 
//                                                  std::complex<double> *values){
//    return mkl_sparse_z_create_coo(A,indexing,rows,cols,nnz,row_indx,col_indx,values);    
//}
//
//template<typename DATATYPE>
//sparse_status_t sparse_mv(const sparse_operation_t operation, const DATATYPE alpha, const sparse_matrix_t A, const struct matrix_descr descr, const DATATYPE *x, const DATATYPE beta, DATATYPE *y);
//
//template<>
//sparse_status_t sparse_mv<double>(const sparse_operation_t operation, const double alpha, const sparse_matrix_t A, const struct matrix_descr descr, const double *x, const double beta, double *y){
//    return mkl_sparse_d_mv(operation,alpha,A,descr,x,beta,y);    
//}
//template<>
//sparse_status_t sparse_mv<std::complex<double> > (const sparse_operation_t operation, const std::complex<double> alpha, const sparse_matrix_t A, const struct matrix_descr descr, const std::complex<double> *x, const std::complex<double> beta, std::complex<double> *y){
//    return mkl_sparse_z_mv (operation,alpha,A,descr,x,beta,y);   
//}
//
//template<typename DATATYPE>
//sparse_status_t sparse_mm(const sparse_operation_t operation, 
//                          const DATATYPE alpha, 
//                          const sparse_matrix_t A, 
//                          const struct matrix_descr descr, 
//                          const sparse_layout_t layout, 
//                          const DATATYPE* B,
//                          const int columns, 
//                          const int ldb, 
//                          const DATATYPE beta, 
//                          DATATYPE* C, 
//                          const int ldc);
//
//template<>
//sparse_status_t sparse_mm<double>(const sparse_operation_t operation,  
//                                  const double alpha,
//                                  const sparse_matrix_t A,
//                                  const struct matrix_descr descr,
//                                  const sparse_layout_t layout,
//                                  const double* B,
//                                  const int columns,
//                                  const int ldb,
//                                  const double beta,
//                                  double* C,
//                                  const int ldc){
//    return mkl_sparse_d_mm(operation,alpha,A,descr,layout,B,columns,ldb,beta,C,ldc);    
//}
//template<>
//sparse_status_t sparse_mm<std::complex<double> > (const sparse_operation_t operation, 
//                                                  const std::complex<double> alpha,
//                                                  const sparse_matrix_t A,
//                                                  const struct matrix_descr descr,
//                                                  const sparse_layout_t layout,
//                                                  const std::complex<double>* B,
//                                                  const int columns,
//                                                  const int ldb,
//                                                  const std::complex<double> beta,
//                                                  std::complex<double>* C,
//                                                  const int ldc){
//    return mkl_sparse_z_mm (operation,alpha,A,descr,layout,B,columns,ldb,beta,C,ldc);   
//}
//
//sparse_status_t destroy(sparse_matrix_t A){
//    return mkl_sparse_destroy(A);
//}
////////////////////////////////////////////////////////////////////////////////////
//
//// sparse mv
//template <>
//template<typename DATATYPE>
//std::unique_ptr<DenseTensor<1, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul(
//    const SparseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL>& mat,
//    const DenseTensor<1, DATATYPE, MTYPE::, DEVICETYPE::SYCL>& vec,
//    TRANSTYPE trans)
//{
//    sparse_matrix_t cooA;
//    struct matrix_descr descrA;
//    sparse_status_t status;
//
//    auto shape = mat.ptr_map->get_global_shape();
//    auto nnz = mat.get_num_nonzero();
//
//    int num_row;
//    int num_col;
//    int* row_indx;
//    int* col_indx;
//    if(trans == TRANSTYPE::N){
//        row_indx = &mat.complete_index[0];
//        col_indx = &mat.complete_index[nnz];
//        num_row = shape[0];
//        num_col = shape[1];
//    }
//    else{
//        row_indx = &mat.complete_index[nnz];
//        col_indx = &mat.complete_index[0];
//        num_row = shape[1];
//        num_col = shape[0];
//    }    
//
//    auto ptr_comm=vec.copy_comm();
//
//    assert (num_col==vec.ptr_map->get_global_shape(0));
//    //std::array<int, 1> output_shape = {static_cast<unsigned long>(num_row)};
//    std::array<int, 1> output_shape = {num_row};
//    std::unique_ptr<Map<1,MTYPE::>> ptr_output_map = std::make_unique<Contiguous1DColMap<1>> (output_shape, 0,1);
//
//    std::unique_ptr<DenseTensor<1, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > output = std::make_unique<DenseTensor<1, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>>(ptr_comm, ptr_output_map );
//
//    status = create_coo<DATATYPE>( &cooA,
//                                   SPARSE_INDEX_BASE_ZERO,
//                                   num_row,    // number of rows
//                                   num_col,    // number of cols
//                                   nnz,  // number of nonzeros
//                                   mat.complete_index.get(),
//                                   mat.complete_index.get()+nnz,
//                                   mat.complete_value.get() );
//
//    assert (status == SPARSE_STATUS_SUCCESS);
//
//    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
//    descrA.diag = SPARSE_DIAG_NON_UNIT;
//
//    if(trans == TRANSTYPE::N){
//        status = sparse_mv<DATATYPE>( SPARSE_OPERATION_NON_TRANSPOSE, 1.0, cooA, descrA, vec.data.get(), 0.0, output->data.get());
//    }
//    else{
//        status = sparse_mv<DATATYPE>( SPARSE_OPERATION_TRANSPOSE,     1.0, cooA, descrA, vec.data.get(), 0.0, output->data.get());
//    }
//    assert (status == SPARSE_STATUS_SUCCESS);
//    status = mkl_sparse_destroy(cooA);
//    assert (status == SPARSE_STATUS_SUCCESS);
//
//    return output;
//}
//
//// sparse mm 
//template <>
//template<typename DATATYPE>
//std::unique_ptr< DenseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::matmul(
//    const SparseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL>& mat1,
//    const DenseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL>& mat2,
//    TRANSTYPE trans1,
//    TRANSTYPE trans2)
//{
//    // the code for the case with trans2 ==TRANSTYPE::Y is not yet developed
//    assert (trans2 == TRANSTYPE::N);
//    sparse_matrix_t cooA;
//    struct matrix_descr descrA;
//    sparse_status_t status;
//
//    auto shape = mat1.ptr_map->get_global_shape();
//    auto nnz = mat1.get_num_nonzero();
//
//    int num_row;
//    int num_col;
//    int* row_indx;
//    int* col_indx;
//
//    if(trans1 == TRANSTYPE::N){
//        row_indx = &mat1.complete_index[0];
//        col_indx = &mat1.complete_index[nnz];
//        num_row = shape[0];
//        num_col = shape[1];
//    }
//    else{
//        row_indx = &mat1.complete_index[nnz];
//        col_indx = &mat1.complete_index[0];
//        num_row = shape[1];
//        num_col = shape[0];
//    }    
//
//    auto ptr_comm=mat2.copy_comm();
//    
//    assert (num_col==mat2.ptr_map->get_global_shape(0));
//    //std::array<int, 2> output_shape = {static_cast<unsigned long>(num_row), mat2.ptr_map->get_global_shape(1)};
//    std::array<int, 2> output_shape = {num_row, mat2.ptr_map->get_global_shape(1)};
//    std::unique_ptr<Map<2,MTYPE::>> ptr_output_map = std::make_unique<Contiguous1DColMap<2>> (output_shape, 0,1);
//
//    std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > output = std::make_unique<DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > (ptr_comm, ptr_output_map );
//
//    status = create_coo<DATATYPE>( &cooA,
//                                   SPARSE_INDEX_BASE_ZERO,
//                                   num_row,    // number of rows
//                                   num_col,    // number of cols
//                                   nnz, // number of nonzeros
//                                   mat1.complete_index.get(),
//                                   mat1.complete_index.get()+nnz,
//                                   mat1.complete_value.get() );
//
//    assert (status == SPARSE_STATUS_SUCCESS);
//
//    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
//    descrA.diag = SPARSE_DIAG_NON_UNIT;
////sparse_status_t mkl_sparse_d_mm (const sparse_operation_t operation, const DATATYPE alpha, const sparse_matrix_t A, const struct matrix_descr descr, const sparse_layout_t layout, const DATATYPE *B, const SYCL_INT columns, const SYCL_INT ldb, const DATATYPE beta, DATATYPE *C, const SYCL_INT ldc);
//
//    if(trans1 == TRANSTYPE::N){
//        status = sparse_mm<DATATYPE>( SPARSE_OPERATION_NON_TRANSPOSE, 
//                                      1.0, 
//                                      cooA, 
//                                      descrA, 
//                                      SPARSE_LAYOUT_ROW_MAJOR, 
//                                      mat2.data.get(), 
//                                      mat2.ptr_map->get_global_shape(1), 
//                                      mat2.ptr_map->get_global_shape(1), 
//                                      0.0, 
//                                      output->data.get(), 
//                                      output->ptr_map->get_global_shape(1));
//    }
//    else{
//        status = sparse_mm<DATATYPE>( SPARSE_OPERATION_TRANSPOSE,     
//                                      1.0, 
//                                      cooA, 
//                                      descrA, 
//                                      SPARSE_LAYOUT_ROW_MAJOR, 
//                                      mat2.data.get(), 
//                                      mat2.ptr_map->get_global_shape(1), 
//                                      mat2.ptr_map->get_global_shape(1), 
//                                      0.0, 
//                                      output->data.get(), 
//                                      output->ptr_map->get_global_shape(1));
//    }
//    assert (status == SPARSE_STATUS_SUCCESS);
//    status = destroy(cooA);
//    assert (status == SPARSE_STATUS_SUCCESS);
//    return output;
//}


//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template<>
template<typename DATATYPE>
//void SE::TensorOp<MTYPE::,DEVICETYPE::SYCL>::orthonormalize<DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>( 
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::orthonormalize( 
    DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat,  
    const std::string method)
{
    using REALTYPE = typename real_type<DATATYPE>::type;
	using TensorOp = TensorOp<MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>;

    const auto number_of_vectors = mat.ptr_map->get_global_shape(1);
    const auto vector_size       = mat.ptr_map->get_global_shape(0);
    
    if(method == "qr"){
        auto eigvec = mat.copy_data();
        DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> output ( mat.copy_comm(), mat.copy_map() );
        std::unique_ptr<DATATYPE[]> tau(malloc<DATATYPE,DEVICETYPE::SYCL>(number_of_vectors));
        //std::unique_ptr<DATATYPE[]> tau(new DATATYPE[number_of_vectors]);
        int info = geqrf<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, vector_size, number_of_vectors, eigvec.get(), vector_size, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        info = orgqr<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, vector_size, number_of_vectors, eigvec.get(), vector_size, tau.get());
        if(info != 0){
            std::cout << "QR decomposition failed!" << std::endl;
            exit(1);
        }
        memcpy<DATATYPE, DEVICETYPE::SYCL>(mat.data.get(), eigvec.get(), number_of_vectors*vector_size);
        
    }
    else{
        auto submatrix = TensorOp::matmul(*TensorOp::conjugate(mat), mat, TRANSTYPE::T, TRANSTYPE::N);
        std::unique_ptr<DATATYPE[]> submatrix_eigvals(malloc<DATATYPE,DEVICETYPE::SYCL>(number_of_vectors));
        //std::unique_ptr<REALTYPE[]> submatrix_eigvals(new REALTYPE[number_of_vectors]);
        syev<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, 'V', 'U', number_of_vectors, submatrix->data.get(), number_of_vectors, submatrix_eigvals.get());

        auto output = TensorOp::matmul(mat, *submatrix, TRANSTYPE::N, TRANSTYPE::N);
        //vector should be normalized
        for(int i=0; i<number_of_vectors; i++){
            const auto norm = nrm2<DATATYPE, DEVICETYPE::SYCL>(vector_size, &output->data[i], number_of_vectors);
            assert(norm != 0.0);
            scal<REALTYPE, DATATYPE, DEVICETYPE::SYCL>(vector_size, 1.0 / norm, &output->data[i], number_of_vectors);
        }
        memcpy<DATATYPE, DEVICETYPE::SYCL>(mat.data.get(), output->data.get(), number_of_vectors*vector_size);
        
    }
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, const DATATYPE* scale_coeff){
    if(scale_coeff == nullptr){
        std::cout << "WRONG scale_coeff in TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors!" << std::endl;
        exit(1);
    }
    int vec_size = mat.ptr_map->get_global_shape()[0];
    int num_vec = mat.ptr_map->get_global_shape()[1];
    for(int index = 0; index < num_vec; index++){
        scal<DATATYPE,DATATYPE,DEVICETYPE::SYCL>(vec_size, scale_coeff[index], &mat.data[index], num_vec);
    }
    return;
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, const DATATYPE scale_factor){
    scal<DATATYPE, DATATYPE, DEVICETYPE::SYCL>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(), 1);
    return;
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, const typename real_type<DATATYPE>::type* scale_coeff){
    using REALTYPE = typename real_type<DATATYPE>::type;
    if(scale_coeff == nullptr){
        std::cout << "WRONG scale_coeff in TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors!" << std::endl;
        exit(1);
    }
    int vec_size = mat.ptr_map->get_global_shape()[0];
    int num_vec = mat.ptr_map->get_global_shape()[1];
    for(int index = 0; index < num_vec; index++){
        scal<REALTYPE,DATATYPE,DEVICETYPE::SYCL>(vec_size, scale_coeff[index], &mat.data[index], num_vec);
    }
    return;
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, const typename real_type<DATATYPE>::type scale_factor){
    using REALTYPE = typename real_type<DATATYPE>::type;
    scal<REALTYPE, DATATYPE, DEVICETYPE::SYCL>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(), 1);
    return;
}


//X + bY
template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::add_(
            const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat1,
            const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat2, const typename real_type<DATATYPE>::type coeff2){
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    assert(mat1.ptr_map->get_global_shape()[1] == mat2.ptr_map->get_global_shape()[1]);
    axpy<DATATYPE, DEVICETYPE::SYCL>(mat1.ptr_map->get_global_shape()[0]*mat1.ptr_map->get_global_shape()[1],coeff2,mat2.data.get(),1,mat1.data.get(),1);
    return;
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::get_norm_of_vectors(const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, typename real_type<DATATYPE>::type* norm, const int norm_size, const bool root /*=true*/){
    assert(mat.ptr_map->get_global_shape()[1] >= norm_size);
    for(int i=0;i<norm_size;i++){
        norm[i] = nrm2<DATATYPE, DEVICETYPE::SYCL>(mat.ptr_map->get_global_shape()[0], &mat.data[i], mat.ptr_map->get_global_shape()[1]);
        if (root==false){
            norm[i] = norm[i]*norm[i];  
        }
    }
    return;
}

//norm_i = ||A*B|| (i=0~norm_size-1)
template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::vectorwise_dot(const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat1,const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat2,
                               typename real_type<DATATYPE>::type* norm, const int norm_size){

    assert (mat1.ptr_map->get_global_shape() == mat2.ptr_map->get_global_shape());
    assert (mat1.ptr_map->get_local_shape() == mat2.ptr_map->get_local_shape());
    assert (mat1.ptr_map->get_global_shape()[1] >= norm_size);


    const int vec_size = mat1.ptr_map->get_local_shape()[0];
    const int num_vec  = mat1.ptr_map->get_local_shape()[1]; 

    const int local_size =mat1.ptr_map->get_num_local_elements();
    DATATYPE* buff = malloc<DATATYPE,DEVICETYPE::SYCL>(local_size);
    auto mat = SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::conjugate(mat1);
    vMul<DATATYPE,DEVICETYPE::SYCL>(local_size, mat->data.get(), mat2.data.get(), buff);

    std::fill_n(norm, num_vec, 0.0);
    
    for(int i=0;i<norm_size;i++){
        #pragma omp parallel for reduction(+:norm[i]) 
        for (int j=0; j<vec_size; j++){
            //std::array<int, 2> arr = {j,i};
            const auto index = mat1.ptr_map->unpack_local_array_index({j,i});
            norm[i] += std::real(buff[index]);
        }

    }
    free<DEVICETYPE::SYCL>(buff);
    return;
}

template <>
template<typename DATATYPE>
void SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::copy_vectors(
        DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat1,
        const DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat2, const int new_size){
    assert(mat1.ptr_map->get_global_shape()[1] >= new_size);
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    int vec_size = mat1.ptr_map->get_global_shape()[0];
    for(int i=0;i<new_size;i++){
        copy<DATATYPE, DEVICETYPE::SYCL>(vec_size, &mat2.data[i], mat2.ptr_map->get_global_shape()[1], &mat1.data[i], mat1.ptr_map->get_global_shape()[1]);
    }
    return;
}

template <>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::append_vectors(
        DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat1,
        DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat2){
    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);

    std::array<int, 2> new_shape = {mat1.ptr_map->get_global_shape()[0], mat1.ptr_map->get_global_shape()[1] + mat2.ptr_map->get_global_shape()[1]};
    std::unique_ptr<Map<2,MTYPE::Contiguous1DCol> > ptr_new_map = std::make_unique<Contiguous1DColMap<2> >(new_shape, mat1.ptr_comm->get_rank(), mat1.ptr_comm->get_world_size());
    auto mat = std::make_unique< DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > (mat1.copy_comm(), ptr_new_map);
    //auto ptr_mat =std::make_unique<DenseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > (mat1.copy_comm(), ptr_new_map);
    for(int i=0;i<mat1.ptr_map->get_global_shape()[1];i++){
        copy<DATATYPE, DEVICETYPE::SYCL>(new_shape[0], &mat1.data[i], mat1.ptr_map->get_global_shape()[1], &(mat->data[i]), new_shape[1]);
    }
    for(int i=0;i<mat2.ptr_map->get_global_shape()[1];i++){
        copy<DATATYPE, DEVICETYPE::SYCL>(new_shape[0], &mat2.data[i], mat2.ptr_map->get_global_shape()[1], &(mat->data[i+mat1.ptr_map->get_global_shape()[1]]), new_shape[1]);
    }
    return mat;
}


 // return eigvec
template<>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL> > SE::TensorOp<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>::diagonalize(
        DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>& mat, typename real_type<DATATYPE>::type* eigval){
    // input should be squar matrix 
    assert(mat.ptr_map->get_global_shape()[0] == mat.ptr_map->get_global_shape()[1]); 
    const int block_size = mat.ptr_map->get_global_shape()[0]; 
    auto eigvec = mat.clone(); //std::make_unique<DenseTensor<2, DATATYPE, MTYPE::, DEVICETYPE::SYCL> > (mat);
    int info = syev<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, 'V', 'U', block_size, eigvec->data.get(), block_size, eigval);
    if(info !=0){
        std::cout << "subspace_diagonalization error!" << std::endl;
        exit(-1);
    }
    return eigvec;
}





}
