#pragma once

#include "../DenseTensor.hpp"
#include "../device/LinearOp.hpp"
#include "../VectorUtility.hpp"
#include "DecomposeResult.hpp"
#include <memory>

namespace SE{

template<MTYPE mtype, DEVICETYPE device>
class DirectSolver
{
public:

template<typename DATATYPE>
static
std::unique_ptr<DecomposeResult<DATATYPE, device> >evd(DenseTensor<2,DATATYPE,mtype,device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec){assert(false); return NULL;};
    
};

template<>
class DirectSolver<MTYPE::Contiguous1D,DEVICETYPE::MKL>
{
public:
template<typename DATATYPE>
static
std::unique_ptr<DecomposeResult<DATATYPE, DEVICETYPE::MKL> > evd(DenseTensor<2,DATATYPE,MTYPE::Contiguous1D,DEVICETYPE::MKL>& tensor, DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>* eigvec){

	using REALTYPE = typename real_type<DATATYPE>::type;

    assert(tensor.ptr_map->get_global_shape()[0] == tensor.ptr_map->get_global_shape()[1]);
    const int n = tensor.ptr_map->get_global_shape()[0];
    
    std::unique_ptr<std::complex<REALTYPE>[]> eigvals_ptr(new std::complex<REALTYPE>[n]);
    std::unique_ptr<REALTYPE[]> real_eigvals_ptr(new REALTYPE[n]);
    
    std::unique_ptr<DATATYPE[]> left_eigvec(new DATATYPE[n*n]);
    std::unique_ptr<DATATYPE[]> right_eigvec(new DATATYPE[n*n]);
/*
    double* mat_new = new double[n*n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            //mat_new[i*(n)+j] = 0.0;
            //if(i<n && j<n){
            mat_new[i*n+j] = 1.0;//tensor.data.get()[i*n+j];
            //}
        }
    }
*/
    
    auto mat = tensor.copy_data();
    
    int info = 0;
    info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat.get(), n, eigvals_ptr.get(), left_eigvec.get(), n, right_eigvec.get(), n);
    //info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat_new, n, real_eigvals_ptr,imag_eigvals_ptr, left_eigvec, n, right_eigvec, n);
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }

	#pragma omp parallel for
    for(int i=0; i<n;i++){
		real_eigvals_ptr[i] = std::real(eigvals_ptr[i]);
	}
    eigenvec_sort<DATATYPE, DEVICETYPE::MKL>(real_eigvals_ptr.get(), left_eigvec.get(), n, n);

    //Print eigenvalues
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

    std::vector<REALTYPE> real_eigvals(n);
    std::vector<REALTYPE> imag_eigvals(n);
    for(int i=0;i<n;i++){
		real_eigvals[i] = std::real(eigvals_ptr[i]);	
		imag_eigvals[i] = std::imag(eigvals_ptr[i]);	
	}
    std::unique_ptr<DecomposeResult<DATATYPE,DEVICETYPE::MKL> > return_val = std::make_unique< DecomposeResult<DATATYPE,DEVICETYPE::MKL> >( (int) n, real_eigvals.data(),imag_eigvals.data());
    
    //eigvec = std::move(left_eigvec);
    const int num_guess = eigvec->ptr_map->get_global_shape()[1];
    for(int i=0;i<num_guess;i++){
        copy<DATATYPE, DEVICETYPE::MKL>(n,&left_eigvec.get()[i],n,&eigvec->data[i],num_guess);
    }

    //unique_ptr이니 free 안함    
    //free<DEVICETYPE::MKL>(mat);

    return std::move(return_val);
}
};

template<>
class DirectSolver<MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>
{
public:
//template<typename DATATYPE>
//static
//std::unique_ptr<DecomposeResult<DATATYPE> > evd(DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>& tensor, DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>* eigvec){
//    using REALTYPE = typename real_type<DATATYPE>::type;
//
//    assert(tensor.ptr_map->get_global_shape()[0] == tensor.ptr_map->get_global_shape()[1]);
//    const int n = tensor.ptr_map->get_global_shape()[0];
//    
//    std::complex<REALTYPE>* eigvals_ptr = malloc<std::complex<REALTYPE>, DEVICETYPE::SYCL> (n);
//    //std::unique_ptr<std::complex<REALTYPE>[]> eigvals_ptr(new std::complex<REALTYPE>[n]);
//    std::unique_ptr<REALTYPE[], std::function<void(REALTYPE*)>> real_eigvals_ptr( malloc<DATATYPE,DEVICETYPE::SYCL>(n), free<DEVICETYPE::SYCL>);
//    
//    std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)>> left_eigvec( malloc<DATATYPE,DEVICETYPE::SYCL>( n*n),free<DEVICETYPE::SYCL>  );
//    std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)>> right_eigvec( malloc<DATATYPE,DEVICETYPE::SYCL>(n*n), free<DEVICETYPE::SYCL> );
///*
//    double* mat_new = new double[n*n];
//    for(int i=0;i<n;i++){
//        for(int j=0;j<n;j++){
//            //mat_new[i*(n)+j] = 0.0;
//            //if(i<n && j<n){
//            mat_new[i*n+j] = 1.0;//tensor.data.get()[i*n+j];
//            //}
//        }
//    }
//*/
//    
//    auto mat = tensor.copy_data();
//    
//    int info = 0;
//    info = geev<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, 'V', 'V', n, mat.get(), n, eigvals_ptr, left_eigvec.get(), n, right_eigvec.get(), n);
//    //info = geev<DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, 'V', 'V', n, mat.get(), n, eigvals_ptr.get(), left_eigvec.get(), n, right_eigvec.get(), n);
//    //info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat_new, n, real_eigvals_ptr,imag_eigvals_ptr, left_eigvec, n, right_eigvec, n);
//    if( info > 0 ) {
//        printf( "The algorithm failed to compute eigenvalues.\n" );
//        exit( 1 );
//    }
//
//	#pragma omp parallel for
//    for(int i=0; i<n;i++){
//		real_eigvals_ptr[i] = std::real(eigvals_ptr[i]);
//	}
//    eigenvec_sort<DATATYPE, DEVICETYPE::SYCL>(real_eigvals_ptr.get(), left_eigvec.get(), n, n);
//
//    //Print eigenvalues
//    
//    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
//    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );
//
//    std::vector<REALTYPE> real_eigvals(n);
//    std::vector<REALTYPE> imag_eigvals(n);
//    for(int i=0;i<n;i++){
//		real_eigvals[i] = std::real(eigvals_ptr[i]);	
//		imag_eigvals[i] = std::imag(eigvals_ptr[i]);	
//	}
//    std::unique_ptr<DecomposeResult<DATATYPE> > return_val = std::make_unique< DecomposeResult<DATATYPE> >( (int) n, real_eigvals,imag_eigvals);
//    
//    //eigvec = std::move(left_eigvec);
//    const int num_guess = eigvec->ptr_map->get_global_shape()[1];
//    for(int i=0;i<num_guess;i++){
//        copy<DATATYPE, DEVICETYPE::SYCL>(n,&left_eigvec.get()[i],n,&eigvec->data[i],num_guess);
//    }
//
//    //free<DEVICETYPE::MKL>(mat);
//    free<DEVICETYPE::SYCL>(eigvals_ptr);
//    return std::move(return_val);
//}

template<typename DATATYPE>
static
std::unique_ptr<DecomposeResult<DATATYPE, DEVICETYPE::SYCL> > evd(DenseTensor<2,DATATYPE,MTYPE::Contiguous1DCol,DEVICETYPE::SYCL>& tensor, DenseTensor<2, DATATYPE, MTYPE::Contiguous1DCol, DEVICETYPE::SYCL>* eigvec){
    using REALTYPE = typename real_type<DATATYPE>::type;
    int n = tensor.ptr_map->get_global_shape()[0];
    assert (n==tensor.ptr_map->get_global_shape()[1]);
    assert (n==eigvec->ptr_map->get_global_shape()[0]);

    auto mat = tensor.copy_data();
    std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)>> eigval( malloc<DATATYPE,DEVICETYPE::SYCL>( n),free<DEVICETYPE::SYCL>  );
    std::unique_ptr<REALTYPE[], std::function<void(REALTYPE*)>> real_eigvals( malloc<DATATYPE,DEVICETYPE::SYCL>(n), free<DEVICETYPE::SYCL>);
    std::unique_ptr<REALTYPE[], std::function<void(REALTYPE*)>> imag_eigvals( malloc<DATATYPE,DEVICETYPE::SYCL>(n), free<DEVICETYPE::SYCL>);

    // output of syev function is already ordered in ascending manner
    syev< DATATYPE, DEVICETYPE::SYCL>(ORDERTYPE::COL, 'V',  'U',n, mat.get(), n, eigval.get() ); 

    memcpy<DATATYPE, DEVICETYPE::SYCL>(eigvec->data.get(), mat.get() , eigvec->ptr_map->get_global_shape()[1]*n );
    if constexpr (is_complex_v<DATATYPE>) {
        for (int i=0; i<n; i++){
            real_eigvals[i] = eigval[i].real;
            imag_eigvals[i] = eigval[i].imag;
        }
    }
    else{
        for (int i=0; i<n; i++){
            real_eigvals[i] = eigval[i];
            imag_eigvals[i] = 0;
        }
    }
    return std::make_unique< DecomposeResult<DATATYPE, DEVICETYPE::SYCL> >( n, real_eigvals.get(),imag_eigvals.get());
}    
};
}

