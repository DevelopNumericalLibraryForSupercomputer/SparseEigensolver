#pragma once
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
#include "../device/MKL/MKLComm.hpp"
#include "../device/MPI/MPIComm.hpp"
//#include "../device/CUDA/CUDAComm.hpp"
#include <memory>

namespace SE{

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::evd(){

    const int n = shape[0];
    std::unique_ptr<double[]> real_eigvals(new double[n]);
    std::unique_ptr<double[]> imag_eigvals(new double[n]);

    assert(shape[0] == shape[1]);

    auto eigvec_0 = new double[n*n];
    auto eigvec_1 = new double[n*n];
    
    //return_val.factor_matrix_sizes[0] = return_val.factor_matrix_sizes[1] = std::make_pair(n,n);
            
    //print_matrix(*this);
    //double wr[n]; //eigenvalues
    //double wi[n]; //complex part of eigenvalues

    int info = geev<double, computEnv::MKL>(ColMajor, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0, n, eigvec_1, n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    //Print eigenvalues
    //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
    
    //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
    //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

    delete eigvec_0;
    delete eigvec_1;

    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (size_t) n,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
}

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> >::evd(){
    const int n = shape[0];
    std::unique_ptr<double[]> real_eigvals(new double[n]);
    std::unique_ptr<double[]> imag_eigvals(new double[n]);
    assert(shape[0] == shape[1]);
    auto eigvec_0 = new double[n*n];
    auto eigvec_1 = new double[n*n];
    int info = geev<double, computEnv::MKL>(ColMajor, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0, n, eigvec_1, n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    delete eigvec_0;
    delete eigvec_1;
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> > >( (size_t) n,std::move(real_eigvals),std::move(imag_eigvals));
    return std::move(return_val);
}

/*
template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::CUDA>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::CUDA>, ContiguousMap<2> >::evd(){
    const int n = shape[0];
    std::unique_ptr<double[]> real_eigvals(new double[n]);
    std::unique_ptr<double[]> imag_eigvals(new double[n]);
    assert(shape[0] == shape[1]);
    auto eigvec_0 = new double[n*n];
    auto eigvec_1 = new double[n*n];
    int info = geev<double, computEnv::CUDA>(ColMajor, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(), eigvec_0, n, eigvec_1, n);
    //Check for convergence
    if( info > 0 ) {
        printf( "The algorithm failed to compute eigenvalues.\n" );
        exit( 1 );
    }
    delete eigvec_0;
    delete eigvec_1;
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::CUDA>, ContiguousMap<2> > >( (size_t) n,std::move(real_eigvals),std::move(imag_eigvals));
    return std::move(return_val);
}
*/

}

