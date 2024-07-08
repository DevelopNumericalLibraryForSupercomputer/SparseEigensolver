#pragma once
#include <string>
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"

namespace SE{

namespace TensorOp {

//mv
template <typename DATATYPE, MTYPE mtype1, MTYPE mtype2, DEVICETYPE device>
DenseTensor<1, DATATYPE, mtype2, device> matmul(
    const DenseTensor<2, DATATYPE, mtype1, device>& mat,
    const DenseTensor<1, DATATYPE, mtype2, device>& vec,
    TRANSTYPE trans=TRANSTYPE::N);
// spmv
template <typename DATATYPE, MTYPE mtype1, MTYPE mtype2, DEVICETYPE device>
DenseTensor<1, DATATYPE, mtype2, device> matmul(
    const SparseTensor<2, DATATYPE, mtype1, device>& mat,
    const DenseTensor <1, DATATYPE, mtype2, device>& vec,
    TRANSTYPE trans=TRANSTYPE::N);

//mm
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device> matmul(
    const DenseTensor<2, DATATYPE, mtype, device>& mat1,
    const DenseTensor<2, DATATYPE, mtype, device>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N, 
    TRANSTYPE trans2=TRANSTYPE::N);
// spmm
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device> matmul(
    const SparseTensor<2, DATATYPE, mtype, device>& mat1,
    const DenseTensor <2, DATATYPE, mtype, device>& mat2,
    TRANSTYPE trans1=TRANSTYPE::N,
    TRANSTYPE trans2=TRANSTYPE::N);
    

//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void orthonormalize(DenseTensor<2, DATATYPE, mtype, device>& mat, std::string method);

//y_i = scale_coeff_i * x_i
template <typename DATATYPE, MTYPE mtype1, DEVICETYPE device>
void scale_vectors(DenseTensor<2, DATATYPE, mtype1, device>& mat, DATATYPE* scale_coeff);

//mat1 + coeff2 * mat2
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device> add( 
                 DenseTensor<2, DATATYPE, mtype, device>& mat1,
                 DenseTensor<2, DATATYPE, mtype, device>& mat2, DATATYPE coeff2);

//norm_i = ||mat_i|| (i=0~norm_size-1)
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void get_norm_of_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat,
                         DATATYPE* norm, const int norm_size);

//mat1_i = mat2_i (i=0~new-1)
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void copy_vectors(
        DenseTensor<2, DATATYPE, mtype, device>& mat1,
        DenseTensor<2, DATATYPE, mtype, device>& mat2, int new_size);

//new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device> append_vectors(
        DenseTensor<2, DATATYPE, mtype, device>& mat1,
        DenseTensor<2, DATATYPE, mtype, device>& mat2);

 // return eigvec
template <typename DATATYPE, MTYPE mtype1, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype1, device> diagonalize(DenseTensor<2, DATATYPE, mtype1, device>& mat, DATATYPE* eigval);



}

}
