#pragma once
#include <string>
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"

namespace SE{

namespace TensorOp {



//mv
template <typename DATATYPE, typename MAPTYPE1, typename MAPTYPE2, DEVICETYPE device>
DenseTensor<1, DATATYPE, MAPTYPE2, device> matmul(
    const DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat,
    const DenseTensor<1, DATATYPE, MAPTYPE2, device>& vec,
    TRANSTYPE trans=TRANSTYPE::N);
// spmv
template <typename DATATYPE, typename MAPTYPE1, typename MAPTYPE2, DEVICETYPE device>
DenseTensor<1, DATATYPE, MAPTYPE2, device> matmul(
    const SparseTensor<2, DATATYPE, MAPTYPE1, device>& mat,
    const DenseTensor <1, DATATYPE, MAPTYPE2, device>& vec,
    TRANSTYPE trans=TRANSTYPE::N);

//mm
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> matmul(
    const DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
    const DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N, 
    TRANSTYPE trans2=TRANSTYPE::N);
// spmm
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> matmul(
    const SparseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
    const DenseTensor <2, DATATYPE, MAPTYPE, device>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N,
    TRANSTYPE trans2=TRANSTYPE::N);
    

//QR
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void orthonormalize(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat, std::string method);

}


// dense matrix multiplication 
template <typename DATATYPE, DEVICETYPE device>
DenseTensor<1,DATATYPE,Contiguous1DMap<1>, device> TensorOp::matmul(
//DenseTensor<1,DATATYPE,Contiguous1DMap<1>, device> TensorOp::matmul<DATATYPE, Contiguous1DMap<2>, Contiguous1DMap<1>, device>(
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, device>& mat,
    const DenseTensor<1, DATATYPE, Contiguous1DMap<1>, device>& vec,
    TRANSTYPE trans=TRANSTYPE::N)
{
    assert ( mat.map.get_global_shape(1) == vec.map.get_global_shape(0) );
    DenseTensor<1,DATATYPE,Contiguous1DMap<1>, device> output ( *vec.copy_comm(), *vec.copy_map() );

    size_t m = mat.map.get_global_shape(0);
    size_t k = mat.map.get_global_shape(1);
    if(trans != TRANPOSE::N){
        m= mat.map.get_global_shape(1);
        k= mat.map.get_global_shape(0);
    }
    //mby k * kby n
    gemm<DATATYPE, device>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, k, vec.data, 1, 0.0, output.data, 1);
    return output;
}


template <typename DATATYPE, DEVICETYPE device>
DenseTensor<2,DATATYPE,Contiguous1DMap<2>, device> TensorOp::matmul(
//DenseTensor<2,DATATYPE,Contiguous1DMap<2>, device> TensorOp::matmul<DATATYPE, Contiguous1DMap<2>, device>(
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, device>& mat1,
    const DenseTensor<2, DATATYPE, Contiguous1DMap<2>, device>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N,
    TRANSTYPE trans2=TRANSTYPE::N)
{
    assert ( mat1.map.get_global_shape(1) == mat2.map.get_global_shape(0) );
    DenseTensor<2,DATATYPE,Contiguous1DMap<2>, device> output ( *mat2.copy_comm(), *mat2.copy_map() );

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
    gemm<DATATYPE, device>(ORDERTYPE::ROW, trans1, trans2, m, n, k, 1.0, mat1.data, k, mat2.data, n, 0.0, output.data, n);
    return output;
}
/*
// sparse mv 
template <typename DATATYPE, DEVICETYPE device>
void TensorOP::matmul<DATATYPE, Contiguous1DMap<2>, Contiguous1DMap<1>, STORETYPE::SPARSE, STORETYPE::DENSE, device>(
    const Tensor<2, DATATYPE, Contiguous1DMap<2>, device, STORETYPE::SPARSE>& mat,
    const Tensor<1, DATATYPE, Contiguous1DMap<1>, device, STORETYPE::DENSE>& vec,
    TRANSTYPE trans,
    Tensor<1, DATATYPE, Contiguous1DMap<1>, device, STORETYPE::DENSE>& output )
{

//    std::unique_ptr<int[]> row_indx(new int[data.size()]) ;
//    std::unique_ptr<int[]> col_indx(new int[data.size()]) ;
//
//    for (size_t i=0; i<data.size(); i++){
//        row_indx[i] = data[i].first[i]        
//    }
//    mat.data
//
//    size_t m = mat.map.get_global_shape(0);
//    size_t k = mat.map.get_global_shape(1);
//    if(trans != TRANPOSE::N){
//        m= mat.map.get_global_shape(1);
//        k= mat.map.get_global_shape(0);
//    }

    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != TRANSTYPE::N){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    std::array<size_t,1> return_size = {m};
    double* return_data = malloc<double, DEVICE::MKL>(m);
    memset<double, DEVICE::MKL>(return_data, 0, m);
    
    if(transa == TRANSTYPE::N){
        for(auto entity : a->data){
            return_data[ entity.first[0] ] += entity.second * v->data[ entity.first[1] ];
        }
    }
    else{
        for(auto entity : a->data){
            return_data[ entity.first[1] ] += entity.second * v->data[ entity.first[0] ];
        }
    }
    //MAPTYPE2* return_map = new MAPTYPE2(return_size, 1);
    Tensor<STORETYPE::Dense, double, 1, DEVICE::MKL, MAPTYPE2>* return_mat = new Tensor<STORETYPE::Dense, double, 1, DEVICE::MKL, MAPTYPE2>(a->comm, return_size, return_data);
    return return_mat;
}  
 
template <typename MAPTYPE>
Tensor<STORETYPE::Dense, double, 2, DEVICE::MKL, MAPTYPE>* spmv(Tensor<STORETYPE::COO, double, 2, DEVICE::MKL, MAPTYPE>* a, 
                                                        Tensor<STORETYPE::Dense, double, 2, DEVICE::MKL, MAPTYPE>* v,
                                                        TRANSTYPE transa){
    size_t m = a->shape[0];
    size_t k = a->shape[1];
    if(transa != TRANSTYPE::N){
        m = a->shape[1]; k = a->shape[0];
    }
    assert (v->shape[0] == k);
    size_t number_of_vec = v->shape[1];
    std::array<size_t,2> return_size = {m, number_of_vec};
    double* return_data = malloc<double, DEVICE::MKL>(m*number_of_vec);
    memset<double, DEVICE::MKL>(return_data, 0, m*number_of_vec);
    
    if(transa == TRANSTYPE::N){
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
    //MAPTYPE* return_map = new MAPTYPE(return_size, 1);
    Tensor<STORETYPE::Dense, double, 2, DEVICE::MKL, MAPTYPE>* return_mat = new Tensor<STORETYPE::Dense, double, 2, DEVICE::MKL, MAPTYPE>(a->comm, return_size, return_data);
    return return_mat;
}   




*/

}
