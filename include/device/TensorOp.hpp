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
    

//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void orthonormalize(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat, std::string method);


template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
void scale_vectors(DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat, DATATYPE* scale_coeff);

template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> add( 
                 DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
                 DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2, DATATYPE coeff2);

template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void get_norm_of_vectors(DenseTensor<2, DATATYPE, MAPTYPE, device>& mat,
                         DATATYPE* norm, size_t norm_size);

template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
void copy_vectors(
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2, size_t new_size);


template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> append_vectors(
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat1,
        DenseTensor<2, DATATYPE, MAPTYPE, device>& mat2);

 // return eigvec
template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE1, device> diagonalize(DenseTensor<2, DATATYPE, MAPTYPE1, device>& mat, DATATYPE* eigval);



}

}