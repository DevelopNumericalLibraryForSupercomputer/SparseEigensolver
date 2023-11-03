#pragma once
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
//#include "../device/MKL/MKLComm.hpp"
//#include "../device/MPI/MPIComm.hpp"
//#include "../device/CUDA/CUDAComm.hpp"
#include <memory>

namespace SE{

//template <>
//std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
//        DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::evd(){

template<typename datatype, size_t dimension, typename comm, typename map>
std::unique_ptr<DecomposeResult<datatype, dimension, comm, map> > evd(DenseTensor<datatype, dimension, comm, map>){
    std::cout << "EVD for the rank-" << dimension << " tensor is not implemented.";
    exit(1);
}

template<typename datatype, typename comm, typename map>
std::unique_ptr<DecomposeResult<datatype, 2, comm, map> > evd(DenseTensor<datatype, 2, comm, map> tensor){
    assert(tensor.shape[0] == tensor.shape[1]);
    const size_t n = tensor.shape[0];

    //datatype* real_eigvals_tmp = malloc<datatype, comm>(n);
    //datatype* imag_eigvals_tmp = malloc<datatype, comm>(n);
    //std::unique_ptr<datatype[]> real_eigvals(new double[n]);
    //std::unique_ptr<datatype[]> imag_eigvals(new double[n]);
    //auto real_eigvals = std::make_unique<datatype*>(malloc<datatype, >(n));
    //auto imag_eigvals = std::make_unique<datatype*>(malloc<datatype, >(n));
    /*

    auto eigvec_0 = new double[n*n];
    auto eigvec_1 = new double[n*n];
    
    int info = geev<double, computEnv::MKL>(ColMajor, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0, n, eigvec_1, n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    eigenvec_sort<double, computEnv::MKL>(real_eigvals.get(), eigvec_0, n,n);
    //Print eigenvalues
    //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

    delete eigvec_0;
    delete eigvec_1;
    */
    auto return_val = std::make_unique< DecomposeResult<datatype, 2, comm, map >( (size_t) n, std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
}


}

