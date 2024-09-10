#pragma once

#include "../DenseTensor.hpp"
#include "../device/LinearOp.hpp"
#include "../VectorUtility.hpp"
#include "DecomposeResult.hpp"
#include <memory>

namespace SE{

// Dummy 
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device >
std::unique_ptr<DecomposeResult<DATATYPE> > evd(DenseTensor<2,DATATYPE,mtype,device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec){assert(true); return NULL;};

template <typename DATATYPE, MTYPE mtype>
std::unique_ptr<DecomposeResult<DATATYPE> > evd(DenseTensor<2,DATATYPE,mtype,DEVICETYPE::MKL>& tensor, DenseTensor<2, DATATYPE, mtype, DEVICETYPE::MKL>* eigvec){
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
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val = std::make_unique< DecomposeResult<DATATYPE> >( (int) n, real_eigvals,imag_eigvals);
    
    //eigvec = std::move(left_eigvec);
    const int num_guess = eigvec->ptr_map->get_global_shape()[1];
    for(int i=0;i<num_guess;i++){
        copy<DATATYPE, DEVICETYPE::MKL>(n,&left_eigvec.get()[i],n,&eigvec->data[i],num_guess);
    }

    //unique_ptr이니 free 안함    
    //free<DEVICETYPE::MKL>(mat);

    return std::move(return_val);
}


}

