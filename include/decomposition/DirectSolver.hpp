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
    assert(tensor.ptr_map->get_global_shape()[0] == tensor.ptr_map->get_global_shape()[1]);
    const int n = tensor.ptr_map->get_global_shape()[0];
    
    //std::unique_ptr<DATATYPE[]> real_eigvals_ptr(new DATATYPE[n]);
    //std::unique_ptr<DATATYPE[]> imag_eigvals_ptr(new DATATYPE[n]);
    
    //std::unique_ptr<DATATYPE[]> eigvec_0(new DATATYPE[n*n]);
    //std::unique_ptr<DATATYPE[]> eigvec_1(new DATATYPE[n*n]);
    
    printf( "direct5.\n" );
    double* mat_new = new double[n*n];
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            //mat_new[i*(n)+j] = 0.0;
            //if(i<n && j<n){
            mat_new[i*n+j] = 1.0;//tensor.data.get()[i*n+j];
            //}
        }
    }
    double* real_eigvals_ptr = new double[n];
    double* imag_eigvals_ptr = new double[n];
    double* eigvec_0 = new double[n*n];
    double* eigvec_1 = new double[n*n];

    printf( "direct5.\n" );
    
    //auto mat = tensor.copy_data();
    
    
    int info = 0;
    //info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat_new, n, real_eigvals_ptr.get(),imag_eigvals_ptr.get(), eigvec_0.get(), n, eigvec_1.get(), n);
    info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat_new, n, real_eigvals_ptr,imag_eigvals_ptr, eigvec_0, n, eigvec_1, n);
    printf( "INFO = %d\n",info);
    //printf( "eigvals = %f%f%f\n",real_eigvals_ptr.get()[0], real_eigvals_ptr.get()[1], real_eigvals_ptr.get()[2]);
    printf( "direct5.\n" );
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    
    delete[] mat_new;
    delete[] real_eigvals_ptr;
    delete[] imag_eigvals_ptr;
    delete[] eigvec_0;
    delete[] eigvec_1;
    
    //eigenvec_sort<DATATYPE, DEVICETYPE::MKL>(real_eigvals_ptr.get(), eigvec_0.get(), n, n);
    printf( "direct6.\n" );
    //Print eigenvalues
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );
    /*
    std::vector<DATATYPE> real_eigvals(real_eigvals_ptr.get(), real_eigvals_ptr.get()+n);
    std::vector<DATATYPE> imag_eigvals(imag_eigvals_ptr.get(), imag_eigvals_ptr.get()+n);
    
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val = std::make_unique< DecomposeResult<DATATYPE> >( (int) n, real_eigvals,imag_eigvals);
    
    eigvec = std::move(eigvec_0);
    const int num_guess = eigvec->ptr_map->get_global_shape()[1];
    for(int i=0;i<num_guess;i++){
        copy<DATATYPE, DEVICETYPE::MKL>(n,&eigvec_0.get()[i],n,&eigvec->data[i],num_guess);
    }

    //unique_ptr이니 free 안함    
    //free<DEVICETYPE::MKL>(mat);

    return std::move(return_val);
    */
    return NULL;
}


}

