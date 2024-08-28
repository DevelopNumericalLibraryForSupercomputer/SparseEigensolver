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
    

//template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//DenseTensor<2, DATATYPE, mtype, device> sum( const DenseTensor<2, DATATYPE, mtype, device>& mat1, const DATATYPE alpha,const DenseTensor<2, DATATYPE, mtype, device>& mat2, const DATATYPE beta);


//Orthonormalization
//n vectors with size m should be stored in m by n matrix (row-major).
//Each coulumn correponds to the vector should be orthonormalized.
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void orthonormalize(DenseTensor<2, DATATYPE, mtype, device>& mat, const std::string method);

//y_i = scale_coeff_i * x_i
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE* scale_coeff);

//y_i = scale_coeff_i * x_i
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr<DenseTensor<2,DATATYPE, mtype, device>> scale_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE* scale_coeff){
	auto output = mat.clone();
	scale_vectors_(*output, scale_coeff);
	return output; 
};


//scale_coeff * x
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE scale_factor);

//mat1 + coeff2 * mat2
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void add_(const DenseTensor<2, DATATYPE, mtype, device>& mat1,
          const DenseTensor<2, DATATYPE, mtype, device>& mat2, const DATATYPE coeff2);

//mat1 + coeff2 * mat2
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> >add( 
                 const DenseTensor<2, DATATYPE, mtype, device>& mat1,
                 const DenseTensor<2, DATATYPE, mtype, device>& mat2, const DATATYPE coeff2){
	auto output = mat1.clone();
	add_(*output, mat2, coeff2);
	return output;
};


template <int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void conjugate_(const DenseTensor<dimension, DATATYPE, mtype, device>& mat){
	static_assert(false==is_complex_v<DATATYPE>, "conjgate function detect error");
	return;
}
template <int dimension, typename T, MTYPE mtype, DEVICETYPE device>
void conjugate_(const DenseTensor<dimension, std::complex<T>, mtype, device>& mat){
	auto num_local_element = mat.ptr_map->get_num_local_elements();

	#pragma omp parallel for 
	for (int i=0; i<num_local_element; i++){
		mat.data[i] = std::conj(mat.data[i]);
	}
	return;
}

template <int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > conjugate(const DenseTensor<dimension, DATATYPE, mtype, device>& mat){

	auto output = mat.clone();
	conjugate_(*output);
	return output;
}

//norm_i = ||mat_i|| (i=0~norm_size-1)
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void get_norm_of_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat,
                         DATATYPE* norm, const int norm_size, const bool root=true);

//norm_i = ||A*B|| (i=0~norm_size-1)
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void vectorwise_dot(const DenseTensor<2, DATATYPE, mtype, device>& mat1,const DenseTensor<2, DATATYPE, mtype, device>& mat2,
                               DATATYPE* norm, const int norm_size);

//mat1_i = mat2_i (i=0~new-1)
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void copy_vectors(
        DenseTensor<2, DATATYPE, mtype, device>& mat1,
        const DenseTensor<2, DATATYPE, mtype, device>& mat2, int new_size);

//new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > append_vectors(
        DenseTensor<2, DATATYPE, mtype, device>& mat1,
        DenseTensor<2, DATATYPE, mtype, device>& mat2);

// return eigvec
template <typename DATATYPE, MTYPE mtype1, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype1, device> diagonalize(DenseTensor<2, DATATYPE, mtype1, device>& mat, DATATYPE* eigval);



}

}
