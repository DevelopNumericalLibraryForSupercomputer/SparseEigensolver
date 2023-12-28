#pragma once
#include "LinearOp.hpp"
#include "../TensorOp.hpp"
#include "CUDAComm.hpp"
namespace SE{

// dense mv
template <>
DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::CUDA> TensorOp::matmul(
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat,
    const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::CUDA>& vec,
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
    DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::CUDA> output ( *vec.copy_comm(), output_map);
    //mby k * kby 1
    gemm<double, DEVICETYPE::CUDA>(ORDERTYPE::ROW, trans, TRANSTYPE::N, m, 1, k, 1.0, mat.data, mat.map.get_global_shape(1), vec.data, 1, 0.0, output.data, 1);
    return output;
}

// dense mm
template <>
DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::CUDA> TensorOp::matmul(
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat1,
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat2,
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
    DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::CUDA> output ( *mat2.copy_comm(), output_map );
    //mby k * kby n
    gemm<double, DEVICETYPE::CUDA>(ORDERTYPE::ROW, trans1, trans2, m, n, k, 1.0, mat1.data, mat1.map.get_global_shape(1), mat2.data, mat2.map.get_global_shape(1), 0.0, output.data, n);
    return output;
}

// sparse mv
template <>
DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::CUDA> SE::TensorOp::matmul<double, Contiguous1DMap<2>, Contiguous1DMap<1>, DEVICETYPE::CUDA>(
    const SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat,
    const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::CUDA>& vec,
    TRANSTYPE trans)
{
    //not implemented
    return DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::CUDA>(vec);
}

// sparse mm 
template <>
DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA> SE::TensorOp::matmul<double, Contiguous1DMap<2>, DEVICETYPE::CUDA>(
    const SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat1,
    const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat2,
    TRANSTYPE trans1,
    TRANSTYPE trans2)
{
    //not implemented
    return DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>(mat2);
}
//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <>
void SE::TensorOp::orthonormalize<double, Contiguous1DMap<2>, DEVICETYPE::CUDA>( 
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::CUDA>& mat,  
    std::string method)
{
    auto number_of_vectors = mat.map.get_global_shape(1);
    auto vector_size       = mat.map.get_global_shape(0);
    
    if(method == "qr"){
        //not implemented
        exit(-1);
    }
    else{
        std::cout << "default orthonormalization" << std::endl;
        auto submatrix = TensorOp::matmul(mat, mat, TRANSTYPE::T, TRANSTYPE::N);
        std::unique_ptr<double[]> submatrix_eigvals(new double[number_of_vectors]);
        //syev<double, DEVICETYPE::CUDA>(ORDERTYPE::ROW, 'V', 'U', number_of_vectors, submatrix.data, number_of_vectors, submatrix_eigvals.get());

        auto output = TensorOp::matmul(mat, submatrix, TRANSTYPE::N, TRANSTYPE::N);
        //vector should be normalized
        for(size_t i=0; i<number_of_vectors; i++){
            double norm = nrm2<double, DEVICETYPE::CUDA>(vector_size, &output.data[i], number_of_vectors);
            assert(norm != 0.0);
            scal<double, DEVICETYPE::CUDA>(vector_size, 1.0 / norm, &output.data[i], number_of_vectors);
        }
        memcpy<double, DEVICETYPE::CUDA>(mat.data, output.data, number_of_vectors*vector_size);
        //return output;
    }
}
}
