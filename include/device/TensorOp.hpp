#pragma once
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"

namespace SE{

namespace ElapsedTime{
	std::vector<double> dense_matmul_2_1;
	std::vector<double> dense_matmul_2_2;
	std::vector<double> sparse_matmul_2_1;
	std::vector<double> sparse_matmul_2_2;
	std::vector<double> orthonormalize;
	std::vector<double> scale_vectors ;
	std::vector<double> add ;
	std::vector<double> conjugate;
	std::vector<double> norm;
	std::vector<double> vectorwise_dot;
	std::vector<double> copy_vectors;
	std::vector<double> append_vectors;
	std::vector<double> diagonalize;

	void print_one(const std::vector<double>& values, bool detail=false){
		std::cout << values.size() << " " << std::fixed << std::setw(9) << std::setprecision(6) << std::accumulate(values.begin(), values.end(), 0.0) <<std::endl;
		if (detail){
			for (const auto & item : values){
				std::cout << std::scientific << item <<std::endl;
			}
		}
		return;
	}
	void print(bool detail=false){
		std::cout << "======= dense_matmul_2_1: "; print_one(dense_matmul_2_1, detail);
		std::cout << "======= dense_matmul_2_2: "; print_one(dense_matmul_2_2, detail);
		std::cout << "======= sparse_matmul_2_1: "; print_one(sparse_matmul_2_1, detail);
		std::cout << "======= sparse_matmul_2_2: "; print_one(sparse_matmul_2_2, detail);
		std::cout << "======= orthonormalize: "; print_one(orthonormalize, detail);
		std::cout << "======= scale_vectors: "; print_one(scale_vectors, detail);
		std::cout << "======= add: "; print_one(add, detail);
		std::cout << "======= conjugate: "; print_one(conjugate, detail);
		std::cout << "======= norm: "; print_one( norm, detail);
		std::cout << "======= vectorwise_dot: "; print_one( vectorwise_dot, detail);
		std::cout << "======= copy_vectors: "; print_one( copy_vectors, detail);
		std::cout << "======= append_vectors: "; print_one( append_vectors, detail);
		std::cout << "======= diagonalize: "; print_one( diagonalize, detail);
		return;
	}
};

template<MTYPE mtype, DEVICETYPE device>
class TensorOp{
public:

//	std::vector<double> dense_matmul_2_1;
//	std::vector<double> dense_matmul_2_2;
//	std::vector<double> sparse_matmul_2_1;
//	std::vector<double> sparse_matmul_2_2;
//	std::vector<double> orthonormalize;
//	std::vector<double> scale_vectors ;
//	std::vector<double> add ;
//	std::vector<double> conjugate;
//	std::vector<double> norm;
//	std::vector<double> vectorwise_dot;
//	std::vector<double> copy_vectors;
//	std::vector<double> append_vectors;
//	std::vector<double> diagonalize;
//	void print_one(const std::vector<double>& values, bool detail=false){
//		std::cout << values.size() << " " << std::fixed << std::setw(9) << std::setprecision(6) << std::accumulate(values.begin(), values.end(), 0.0) <<std::endl;
//		if (detail){
//			for (const auto & item : values){
//				std::cout << std::scientific << item <<std::endl;
//			}
//		}
//		return;
//	}
//	void print(bool detail=false){
//		std::cout << "======= dense_matmul_2_1: "; print_one(dense_matmul_2_1, detail);
//		std::cout << "======= dense_matmul_2_2: "; print_one(dense_matmul_2_2, detail);
//		std::cout << "======= sparse_matmul_2_1: "; print_one(sparse_matmul_2_1, detail);
//		std::cout << "======= sparse_matmul_2_2: "; print_one(sparse_matmul_2_2, detail);
//		std::cout << "======= orthonormalize: "; print_one(orthonormalize, detail);
//		std::cout << "======= scale_vectors: "; print_one(scale_vectors, detail);
//		std::cout << "======= add: "; print_one(add, detail);
//		std::cout << "======= conjugate: "; print_one(conjugate, detail);
//		std::cout << "======= norm: "; print_one( norm, detail);
//		std::cout << "======= vectorwise_dot: "; print_one( vectorwise_dot, detail);
//		std::cout << "======= copy_vectors: "; print_one( copy_vectors, detail);
//		std::cout << "======= append_vectors: "; print_one( append_vectors, detail);
//		std::cout << "======= diagonalize: "; print_one( diagonalize, detail);
//		return;
//	}
    //mv
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<1, DATATYPE, mtype, device> > matmul(
        const DenseTensor<2, DATATYPE, mtype, device>& mat,
        const DenseTensor<1, DATATYPE, mtype, device>& vec,
        TRANSTYPE trans=TRANSTYPE::N);
    // spmv
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<1, DATATYPE, mtype, device> > matmul(
        const SparseTensor<2, DATATYPE, mtype, device>& mat,
        const DenseTensor <1, DATATYPE, mtype, device>& vec,
        TRANSTYPE trans=TRANSTYPE::N);
    
    //mm
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > matmul(
        const DenseTensor<2, DATATYPE, mtype, device>& mat1,
        const DenseTensor<2, DATATYPE, mtype, device>& mat2,
        TRANSTYPE trans1=TRANSTYPE::N, 
        TRANSTYPE trans2=TRANSTYPE::N);
    // spmm
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > matmul(
        const SparseTensor<2, DATATYPE, mtype, device>& mat1,
        const DenseTensor <2, DATATYPE, mtype, device>& mat2,
        TRANSTYPE trans1=TRANSTYPE::N,
        TRANSTYPE trans2=TRANSTYPE::N);
        
    
    //template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
    //DenseTensor<2, DATATYPE, mtype, device> sum( const DenseTensor<2, DATATYPE, mtype, device>& mat1, const DATATYPE alpha,const DenseTensor<2, DATATYPE, mtype, device>& mat2, const DATATYPE beta);
    
    
    //Orthonormalization
    //n vectors with size m should be stored in m by n matrix (row-major).
    //Each coulumn correponds to the vector should be orthonormalized.
    template <typename DATATYPE>
    static void orthonormalize(DenseTensor<2, DATATYPE, mtype, device>& mat, const std::string method);


    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////// scale function 
    //y_i = scale_coeff_i * x_i
    template <typename DATATYPE>
    static void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE* scale_coeff);

    template <typename DATATYPE>
    static void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const typename real_type<DATATYPE>::type* scale_coeff);

    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<2,DATATYPE, mtype, device>> scale_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE* scale_coeff){
    	auto output = mat.clone();
    	scale_vectors_(*output, scale_coeff);
    	return output; 
    };
    
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<2,DATATYPE, mtype, device>> scale_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat, const typename real_type<DATATYPE>::type* scale_coeff){
    	auto output = mat.clone();
    	scale_vectors_(*output, scale_coeff);
    	return output; 
    };

    //scale_coeff * x
    template <typename DATATYPE>
    static void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const DATATYPE scale_factor);

    //scale_coeff * x
    template <typename DATATYPE>
    static void scale_vectors_(DenseTensor<2, DATATYPE, mtype, device>& mat, const typename real_type<DATATYPE>::type scale_factor);

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    //mat1 + coeff2 * mat2
    template <typename DATATYPE>
    static void add_(const DenseTensor<2, DATATYPE, mtype, device>& mat1,
              const DenseTensor<2, DATATYPE, mtype, device>& mat2, const typename real_type<DATATYPE>::type coeff2);
    
    //mat1 + coeff2 * mat2
    template <typename DATATYPE>
    static std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> >add( 
                     const DenseTensor<2, DATATYPE, mtype, device>& mat1,
                     const DenseTensor<2, DATATYPE, mtype, device>& mat2, const typename real_type<DATATYPE>::type coeff2){
    	auto output = mat1.clone();
    	add_(*output, mat2, coeff2);
    	return output;
    };
    
    
    template <int dimension, typename DATATYPE>
    static void conjugate_(const DenseTensor<dimension, DATATYPE, mtype, device>& mat){
    	static_assert(false==is_complex_v<DATATYPE>, "conjgate function detect error");
    	return;
    }
    template <int dimension, typename T >
    static void conjugate_(const DenseTensor<dimension, std::complex<T>, mtype, device>& mat){
    	const auto num_local_element = mat.ptr_map->get_num_local_elements();
    
    	#pragma omp parallel for 
    	for (int i=0; i<num_local_element; i++){
    		mat.data[i] = std::conj(mat.data[i]);
    	}
    	return;
    }
    
    template <int dimension, typename DATATYPE>
    static std::unique_ptr< DenseTensor<dimension, DATATYPE, mtype, device> > conjugate(const DenseTensor<dimension, DATATYPE, mtype, device>& mat){
    
    	auto output = mat.clone();
    	conjugate_(*output);
    	return output;
    }
    
    //norm_i = ||mat_i|| (i=0~norm_size-1)
    template <typename DATATYPE>
    static void get_norm_of_vectors(const DenseTensor<2, DATATYPE, mtype, device>& mat,
                             typename real_type<DATATYPE>::type* norm, const int norm_size, const bool root=true);
    
    //norm_i = ||A*B|| (i=0~norm_size-1)
    template <typename DATATYPE>
    static void vectorwise_dot(const DenseTensor<2, DATATYPE, mtype, device>& mat1,const DenseTensor<2, DATATYPE, mtype, device>& mat2,
                                   typename real_type<DATATYPE>::type* norm, const int norm_size);
    
    //mat1_i = mat2_i (i=0~new-1)
    template <typename DATATYPE>
    static void copy_vectors(
            DenseTensor<2, DATATYPE, mtype, device>& mat1,
            const DenseTensor<2, DATATYPE, mtype, device>& mat2, const int new_size);
    
    //new_mat = mat1_0, mat1_1,...,mat1_N, mat2_0,...,mat2_M
    template <typename DATATYPE>
    static std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > append_vectors(
            DenseTensor<2, DATATYPE, mtype, device>& mat1,
            DenseTensor<2, DATATYPE, mtype, device>& mat2);
    
    // return eigvec
    template <typename DATATYPE>
    static std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > diagonalize(DenseTensor<2, DATATYPE, mtype, device>& mat, typename real_type<DATATYPE>::type* eigval);

};

}
