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

//namespace TensorOp{
//const int i_zero = 0, i_one = 1, i_negone = -1;
//int ictxt;  // ictxt is defined in MPIComm.hpp
//int info;
//};
//
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
template <typename DATATYPE>
std::unique_ptr<DenseTensor<1,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI> > TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::matmul(
    const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat,
    const DenseTensor<1, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& vec,
    TRANSTYPE trans)
{

    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  

    const int i_zero = 0, i_one = 1, i_negone = -1;
    const DATATYPE zero = 0.0;
    const DATATYPE one = 1.0;

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


    std::unique_ptr<DenseTensor<1,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI> > out = std::make_unique<DenseTensor<1,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI>>( comm_inp->create_comm(), map_inp->create_map());

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
    p_gemm<DATATYPE>( &trans1, &trans2,
            &m, &i_one, &n, &one, 
            mat.data.get(), &i_one, &i_one, desc1, 
            vec.data.get(), &i_one, &i_one, desc2,
            &zero, 
            out->data.get(), &i_one, &i_one, desc3 );
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::dense_matmul_2_1.push_back( ((DATATYPE)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );

    return out;
    //return std::move(out);
}

template <>
template <typename DATATYPE>
std::unique_ptr<DenseTensor<2,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI> > TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::matmul(
    const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
    const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;

    const DATATYPE zero = 0.0;
    const DATATYPE one = 1.0;

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

    std::unique_ptr<DenseTensor<2,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI> > mat3 = std::make_unique<DenseTensor<2,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI>>( comm_inp->create_comm(),  map_inp->create_map());

    const int lld1 = MAX( mat1.ptr_map->get_local_shape()[0], 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape()[0], 1 );
    const int lld3 = MAX( mat3->ptr_map->get_local_shape()[0], 1 );
               
    descinit( desc1, &row1, &col1, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
    assert (info==0);
    descinit( desc2, &row2, &col2, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
    assert (info==0);
    descinit( desc3, &row3, &col3, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld3, &info );
    assert (info==0);

    auto trans1_ = transtype_to_char(trans1);
    auto trans2_ = transtype_to_char(trans2);
    p_gemm<DATATYPE>( &trans1_, 
                      &trans2_, 
                      &m, &n, &k, &one, 
                      mat1.data.get(), &i_one, &i_one, desc1, 
                      mat2.data.get(), &i_one, &i_one, desc2,
                      &zero, 
                      mat3->data.get(), &i_one, &i_one, desc3 );
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::dense_matmul_2_2.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return mat3;
    //return std::move(mat3);
}

//X + bY
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::add_<DATATYPE>(
            const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
            const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2, const typename real_type<DATATYPE>::type coeff2){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;
    int info;

    assert(mat1.ptr_map->get_global_shape()[0] == mat2.ptr_map->get_global_shape()[0]);
    assert(mat1.ptr_map->get_global_shape()[1] == mat2.ptr_map->get_global_shape()[1]);

    const DATATYPE one = 1.0;
    int desc1[9];
    int desc2[9];

    const int row1= mat1.ptr_map->get_global_shape(0);
    const int col1= mat1.ptr_map->get_global_shape(1);
    const int row2= mat2.ptr_map->get_global_shape(0);
    const int col2= mat2.ptr_map->get_global_shape(1);

    const int lld1 = MAX( mat1.ptr_map->get_local_shape()[0], 1 );
    const int lld2 = MAX( mat2.ptr_map->get_local_shape()[0], 1 );

    const auto block_size = mat1.ptr_map->get_block_size();
    assert (block_size == mat2.ptr_map->get_block_size());

    descinit( desc1, &row1, &col1, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld1, &info );
    assert (info==0);
    descinit( desc2, &row2, &col2, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld2, &info );
    assert (info==0);

    const char trans='N';
    p_geadd<DATATYPE>( &trans, &row1, &col1, &coeff2, mat2.data.get(), &i_one, &i_one, desc2, &one, mat1.data.get(), &i_one, &i_one, desc1 );
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::add.push_back( ((DATATYPE)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
    //return std::move(return_mat);
}

//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//always perform QR decomposition
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::orthonormalize(DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const std::string method){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;
    int info;
    //const DATATYPE one = 1.0;
    int desc[9];
    const int row= mat.ptr_map->get_global_shape(0);
    const int col= mat.ptr_map->get_global_shape(1);
    const auto block_size = mat.ptr_map->get_block_size();
    const int lld = MAX( mat.ptr_map->get_local_shape()[0], 1 );
    descinit( desc, &row, &col, &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
    assert (info==0);

    auto m = mat.ptr_map->get_global_shape(0);
    auto n = mat.ptr_map->get_global_shape(1);

    int lwork = -1;
    std::vector<DATATYPE> work(1,0.0);

    DATATYPE* tau =   malloc<DATATYPE,DEVICETYPE::MPI>( MIN(m,n)); //  instead new keyword, malloc<> is required 

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
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::orthonormalize.push_back( ((DATATYPE)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );

    return;

}

//y_i = scale_coeff_i * x_i
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::scale_vectors_(DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const DATATYPE* scale_coeff){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  

    for (int i=0; i< mat.ptr_map->get_local_shape(1); i++ ){
        std::array<int,2> local_arr_idx = {0, i};
        auto global_arr_idx = mat.ptr_map->local_to_global(local_arr_idx);
        auto local_idx = mat.ptr_map->unpack_local_array_index(local_arr_idx);
        scal<DATATYPE, DATATYPE, DEVICETYPE::MPI>(mat.ptr_map->get_local_shape(0), scale_coeff[global_arr_idx[1]], &mat.data[local_idx],1);

    }
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::scale_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
}

template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const DATATYPE scale_factor){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    scal<DATATYPE, DATATYPE, DEVICETYPE::MPI>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(),1);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::scale_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
}

//y_i = scale_coeff_i * x_i
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::scale_vectors_(DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const typename real_type<DATATYPE>::type* scale_coeff){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    using REALTYPE = typename real_type<DATATYPE>::type;

    for (int i=0; i< mat.ptr_map->get_local_shape(1); i++ ){
        std::array<int,2> local_arr_idx = {0, i};
        auto global_arr_idx = mat.ptr_map->local_to_global(local_arr_idx);
        auto local_idx = mat.ptr_map->unpack_local_array_index(local_arr_idx);
        scal<REALTYPE, DATATYPE, DEVICETYPE::MPI>(mat.ptr_map->get_local_shape(0), scale_coeff[global_arr_idx[1]], &mat.data[local_idx],1);

    }
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::scale_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
}

template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::scale_vectors_(
            DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, const typename real_type<DATATYPE>::type scale_factor){
    using REALTYPE = typename real_type<DATATYPE>::type;
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    scal<REALTYPE, DATATYPE, DEVICETYPE::MPI>(mat.ptr_map->get_num_local_elements(), scale_factor, mat.data.get(),1);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::scale_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
}

//norm_i = ||mat_i|| (i=0~norm_size-1)
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::get_norm_of_vectors(const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat,
                         typename real_type<DATATYPE>::type* norm, const int norm_size, const bool root){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;
    assert(mat.ptr_map->get_global_shape()[1] >= norm_size);
    const int vec_size = mat.ptr_map->get_local_shape()[0];
    const int num_vec  = mat.ptr_map->get_local_shape()[1]; 
    DATATYPE* local_sum = malloc<DATATYPE,DEVICETYPE::MPI>(num_vec);
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
    gsum2d<DATATYPE>(&ictxt, "C", "1-tree", &i_one, &num_vec, local_sum, &i_one, &i_negone, &i_negone);


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
    gsum2d<DATATYPE>(&ictxt, "R", "1-tree", &i_one, &norm_size, norm, &i_one, &i_negone, &i_negone);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::norm.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
}

//norm_i = ||A*B|| (i=0~norm_size-1)
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::vectorwise_dot(const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2,
                               typename real_type<DATATYPE>::type* norm, const int norm_size){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;

    assert (mat1.ptr_map->get_global_shape() == mat2.ptr_map->get_global_shape());
    assert (mat1.ptr_map->get_local_shape() == mat2.ptr_map->get_local_shape());
    assert (mat1.ptr_map->get_global_shape()[1] >= norm_size);

    const int local_size =mat1.ptr_map->get_num_local_elements();
    auto buff = malloc<DATATYPE, DEVICETYPE::MPI>(local_size);

    // element-wise multiplication
    vdMul(local_size, mat1.data.get(), mat2.data.get(), buff);

    // get norm (almost same to the above function)
    const int vec_size = mat1.ptr_map->get_local_shape()[0];
    const int num_vec  = mat1.ptr_map->get_local_shape()[1]; 
    DATATYPE* local_sum = malloc<DATATYPE,DEVICETYPE::MPI>(num_vec);
    std::fill_n(local_sum, num_vec, 0.0);
            // debug
//          std::memcpy(mat1.data.get(), buff, local_size*sizeof(double));
//          double* norm_        = malloc<double,DEVICETYPE::MPI> ( num_vec );
//          TensorOp::get_norm_of_vectors(mat1, norm_, num_vec);
//          std::cout << std::setprecision(6) << "norm of mul: " << norm_[0] << ", " << norm_[1] << ", " <<norm_[2] <<std::endl;
//          exit(-1);


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
    gsum2d<DATATYPE>(&ictxt, "C", "1-tree", &i_one, &num_vec, local_sum, &i_one, &i_negone, &i_negone);

    for (int i =0; i<num_vec; i++){
        std::array<int, 2> arr_idx ={0,i};
        auto col_ind = mat1.ptr_map->local_to_global(arr_idx)[1];
        if (col_ind>=0 and col_ind<norm_size){
            norm[col_ind] = local_sum[i];   
        }
    }
    free<DEVICETYPE::MPI>(local_sum);
    free<DEVICETYPE::MPI>(buff);
    // summing up accross the processors (broadcast because norm array is initialized as 0)
    gsum2d<DATATYPE>(&ictxt, "R", "1-tree", &i_one, &norm_size, norm, &i_one, &i_negone, &i_negone);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::vectorwise_dot.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;

}


//mat1_i <- mat2_i (i=0~new-1)
template <>
template <typename DATATYPE>
void TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::copy_vectors(
        DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
        const DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2, const int new_size){
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;
    int info;

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

    p_gemr2d<DATATYPE>(&row, &new_size, mat2.data.get(), &i_one, &i_one, desc2, mat1.data.get(), &i_one, &i_one, desc1, &ictxt);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::copy_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return;
    
}

//new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
template<>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI> > TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::append_vectors(
        DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat1,
        DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat2){

    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    const int i_zero = 0, i_one = 1, i_negone = -1;
    int info;

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

    auto ptr_mat3 = std::make_unique< DenseTensor<2,DATATYPE,MTYPE::BlockCycling, DEVICETYPE::MPI> >( comm_inp->create_comm(), map_inp->create_map());
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
    p_gemr2d<DATATYPE>(&row, &col1, mat1.data.get(), &i_one, &i_one, desc1, ptr_mat3->data.get(), &i_one, &i_one, desc3, &ictxt);
    // mat2->mat3
    p_gemr2d<DATATYPE>(&row, &col2, mat2.data.get(), &i_one, &i_one, desc2, ptr_mat3->data.get(), &i_one, &col_idx, desc3, &ictxt);

    //memcpy<double,DEVICETYPE::MPI>(ptr_mat3->data.get(), local_array, ptr_mat3->ptr_map->get_num_local_elements());
    //free<DEVICETYPE::MPI>(local_array);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::append_vectors.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return ptr_mat3;
    //return std::move(mat3);
}

// // return eigvec
template<>
template<typename DATATYPE>
std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI> > TensorOp<MTYPE::BlockCycling, DEVICETYPE::MPI>::diagonalize(DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI>& mat, typename real_type<DATATYPE>::type* eigval){
    
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
    int info;
    const int i_zero = 0, i_one = 1, i_negone = -1;
    assert(mat.ptr_map->get_global_shape()[0] == mat.ptr_map->get_global_shape()[1]);

    // variables
    const auto block_size = mat.ptr_map->get_block_size();
    const auto global_shape = mat.ptr_map->get_global_shape();
    const auto local_shape  = mat.ptr_map->get_local_shape();
    const int lld = MAX(local_shape[0] , 1 );
    //const int lld = MAX( global_shape[0], 1 );
    const auto N = global_shape[0]; 
    const auto nprow = mat.ptr_map->get_nprow();

    // define new matrix for containing eigvec
    auto eigvec = std::make_unique< DenseTensor<2, DATATYPE, MTYPE::BlockCycling, DEVICETYPE::MPI> >(mat);
    if (block_size[0]*nprow[0]>N or block_size[1]*nprow[1]>N){
//      if(mat.ptr_comm->get_rank()==0) std::cout << "serial diagonalization " << N << std::endl;
//
//      const auto num_global_elements = mat.ptr_map->get_num_global_elements();
//      const auto num_local_elements = mat.ptr_map->get_num_local_elements();
//      auto src =  malloc<DATATYPE, DEVICETYPE::MPI>(num_global_elements);
//      auto trg =  malloc<DATATYPE, DEVICETYPE::MPI>(num_global_elements);
//      std::fill_n(src, num_global_elements, 0.0);
//      std::fill_n(trg, num_global_elements, 0.0);
//      #pragma omp parallel for 
//      for (int i =0; i<num_local_elements; i++){
//          const auto global_index = mat.ptr_map->local_to_global(i);
//          src[global_index] = mat.data[i];
//      }
//      mat.ptr_comm->allreduce(src, num_global_elements, trg, OPTYPE::SUM);
//
//        LAPACKE_dsyev(LAPACK_ROW_MAJOR, 
//                    'V', 'U', 
//                    N, trg, N, eigval);
//
//      #pragma omp parallel for 
//      for (int i =0; i< num_local_elements; i++){
//          const auto global_index = eigvec->ptr_map->local_to_global(i);
//          eigvec->data[i] = trg[global_index];
//      }
//      free<DEVICETYPE::MPI> ( src );
//      free<DEVICETYPE::MPI> ( trg );
        assert(false); // debugging
    }
    else{
        if(mat.ptr_comm->get_rank()==0) std::cout << "parallel diagonalization " << global_shape[0] <<"," << global_shape[1] << "  " << local_shape[0] <<"," <<local_shape[1]  << std::endl;
        int desc1[9]; 
        int desc2[9]; 
    
        // desc1 for mat desc2 for eigvec   
        descinit( desc1, &global_shape[0], &global_shape[1], &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
        descinit( desc2, &global_shape[0], &global_shape[1], &block_size[0], &block_size[1], &i_zero, &i_zero, &ictxt, &lld, &info );
    
        int lwork = -1, liwork = -1, lrwork=-1;
        DATATYPE work_query =-1.0;
        typename real_type<DATATYPE>::type rwork_query = -1.0;
        int iwork_query =-1;
    
        // Workspace query for pdsyevd
        p_syevd<DATATYPE>("V", "U", &N, mat.data.get(), &i_one, &i_one, desc1, eigval, eigvec->data.get(), &i_one, &i_one, desc2, &work_query, &lwork, &rwork_query, &lrwork, &iwork_query, &liwork, &info);

        lwork = (int)work_query;
        lrwork = (int) rwork_query;
        liwork = iwork_query;

        std::vector<DATATYPE> work(lwork);
        std::vector<typename real_type<DATATYPE>::type> rwork(lrwork); // not used for double precision but defined anyway.
        std::vector<int> iwork(liwork);
    
        // Compute eigenvalues and eigenvectors
        p_syevd<DATATYPE>("V", "U", &N, mat.data.get(), &i_one, &i_one, desc1, eigval, eigvec->data.get(), &i_one, &i_one, desc2, work.data(), &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, &info);
        assert(info == 0);
    }
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    ElapsedTime::diagonalize.push_back( ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 );
    return eigvec ;
    
}

}
