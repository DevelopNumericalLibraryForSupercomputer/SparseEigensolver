#pragma once

#include "../DenseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"
#include "../Comm.hpp"

#include "../device/LinearOp.hpp"
#include "../device/TensorOp.hpp"
#include "../Utility.hpp"
#include <memory>

namespace SE{
template<STORETYPE storetype, typename DATATYPE, size_t dimension, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > evd(Tensor<storetype, DATATYPE, dimension, device, MAPTYPE>& tensor){
    std::cout << "EVD for the rank-" << dimension << " tensor is not implemented.";
    exit(1);
}

template <typename DATATYPE, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > evd(Tensor<STORETYPE::Dense, DATATYPE, 2, device, MAPTYPE>& tensor){
    assert(tensor.shape[0] == tensor.shape[1]);
    const size_t n = tensor.shape[0];

    if(tensor.map.is_sliced){
        std::cout << "impossible, evd" << std::endl;
        exit(-1);
    }

    //DATATYPE* real_eigvals_tmp = malloc<DATATYPE, comm>(n);
    //DATATYPE* imag_eigvals_tmp = malloc<DATATYPE, comm>(n);
    std::unique_ptr<DATATYPE[]> real_eigvals(new DATATYPE[n]);
    std::unique_ptr<DATATYPE[]> imag_eigvals(new DATATYPE[n]);
    
    std::unique_ptr<DATATYPE[]> eigvec_0(new DATATYPE[n*n]);
    std::unique_ptr<DATATYPE[]> eigvec_1(new DATATYPE[n*n]);
    
    int info = geev<DATATYPE, device>(SE_layout::ColMajor, 'V', 'V', n, tensor.data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0.get(), n, eigvec_1.get(), n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    eigenvec_sort<DATATYPE, device>(real_eigvals.get(), eigvec_0.get(), n, n);
    //Print eigenvalues
    //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val = std::make_unique< DecomposeResult<DATATYPE> >( (size_t) n, std::move(real_eigvals),std::move(imag_eigvals));
    return std::move(return_val);
}


}

