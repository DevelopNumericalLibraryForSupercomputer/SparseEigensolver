#pragma once
#include "../device/MKL/TensorOp.hpp"
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
//#include "../device/MKL/MKLComm.hpp"
//#include "../device/MPI/MPIComm.hpp"
//#include "../device/CUDA/CUDAComm.hpp"
#include <memory>

namespace SE{
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype> > evd(DenseTensor<datatype, dimension, computEnv, maptype>* tensor){
    std::cout << "EVD for the rank-" << dimension << " tensor is not implemented.";
    exit(1);
}

template <typename datatype, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype> > evd(DenseTensor<datatype, 2, computEnv, maptype>* tensor){
    assert(tensor->shape[0] == tensor->shape[1]);
    const size_t n = tensor->shape[0];

    //datatype* real_eigvals_tmp = malloc<datatype, comm>(n);
    //datatype* imag_eigvals_tmp = malloc<datatype, comm>(n);
    std::unique_ptr<datatype[]> real_eigvals(new datatype[n]);
    std::unique_ptr<datatype[]> imag_eigvals(new datatype[n]);
    
    std::unique_ptr<datatype[]> eigvec_0(new datatype[n*n]);
    std::unique_ptr<datatype[]> eigvec_1(new datatype[n*n]);
    
    int info = geev<datatype, computEnv>(SE_layout::ColMajor, 'V', 'V', n, tensor->data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0.get(), n, eigvec_1.get(), n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    eigenvec_sort<datatype, computEnv>(real_eigvals.get(), eigvec_0.get(), n, n);
    //Print eigenvalues
    //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );
    std::unique_ptr<DecomposeResult<datatype> > return_val = std::make_unique< DecomposeResult<datatype> >( (size_t) n, std::move(real_eigvals),std::move(imag_eigvals));
    return std::move(return_val);
}


}

