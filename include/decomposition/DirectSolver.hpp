#pragma once

#include "../DenseTensor.hpp"
#include "../VectorUtility.hpp"
#include <memory>

namespace SE{
template <typename DATATYPE, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > evd(DenseTensor<2,DATATYPE,MAPTYPE,DEVICETYPE::MKL>& tensor, DenseTensor<2, DATATYPE, MAPTYPE, DEVICETYPE::MKL>* eigvec){
    assert(tensor.map.get_global_shape()[0] == tensor.map.get_global_shape()[1]);
    const size_t n = tensor.map.get_global_shape()[1];

    std::unique_ptr<DATATYPE[]> real_eigvals(new DATATYPE[n]);
    std::unique_ptr<DATATYPE[]> imag_eigvals(new DATATYPE[n]);
    
    std::unique_ptr<DATATYPE[]> eigvec_0(new DATATYPE[n*n]);
    std::unique_ptr<DATATYPE[]> eigvec_1(new DATATYPE[n*n]);
    auto mat = tensor.copy_data();
    int info = geev<DATATYPE, DEVICETYPE::MKL>(ORDERTYPE::ROW, 'V', 'V', n, mat, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0.get(), n, eigvec_1.get(), n);
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    eigenvec_sort<DATATYPE, DEVICETYPE::MKL>(real_eigvals.get(), eigvec_0.get(), n, n);
    //Print eigenvalues
    //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val = std::make_unique< DecomposeResult<DATATYPE> >( (size_t) n, std::move(real_eigvals),std::move(imag_eigvals));
    
    //eigvec = std::move(eigvec_0);
    const size_t num_guess = eigvec->map.get_global_shape()[1];
    for(int i=0;i<num_guess;i++){
        copy<DATATYPE, DEVICETYPE::MKL>(n,&eigvec_0.get()[i],n,&eigvec->data[i],num_guess);
    }
    
    free<DEVICETYPE::MKL>(mat);
    return std::move(return_val);
}


}

