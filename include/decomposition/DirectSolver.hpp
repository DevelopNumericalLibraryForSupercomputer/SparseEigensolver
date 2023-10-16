#pragma once
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
#include <memory>

namespace SE{

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<PROTOCOL::SERIAL>, ContiguousMap<2> > > DenseTensor<double, 2, Comm<PROTOCOL::SERIAL>, ContiguousMap<2> >::decompose(const std::string method){

    const int n = shape[0];
    //auto real_eigvals=std::make_unique< std::vector<double> > (n*n);
    //auto imag_eigvals=std::make_unique< std::vector<double> > (n*n);
    std::unique_ptr<double[]> real_eigvals(new double[n]);
    std::unique_ptr<double[]> imag_eigvals(new double[n]);

    if(method.compare("EVD")==0){
        //eigenvalue decomposition, N by N matrix A = Q Lambda Q-1
        //factor matricies : Q, Q
        //factor matrix sizes : N,N / N,N
        //core tensor = Lambda matrix
        std::cout << "EVD Start!" << std::endl;
        
        //https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2023-1/lapack-least-square-eigenvalue-problem-computation.html
        //real symmteric or complex Hermitian
        //  tridiagonal : steqr, stedc
        //  tridiagonal positive-definite : pteqr
        //generalized symmetric-definite
        //  full storage : sygst / hegst
        //???
        //geev for linear problem (computational routines)


        assert(shape[0] == shape[1]);

        //return_val.factor_matrices[0] = new double[n*n];
        //return_val.factor_matrices[1] = new double[n*n];
        auto eigvec_0 = new double[n*n];
        auto eigvec_1 = new double[n*n];
        
        //return_val.factor_matrix_sizes[0] = return_val.factor_matrix_sizes[1] = std::make_pair(n,n);
                
        //print_matrix(*this);
        //double wr[n]; //eigenvalues
        //double wi[n]; //complex part of eigenvalues

        //lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda,
        //                           double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr) 

        //int info = LAPACKE_dgeev( LAPACK_COL_MAJOR, 'V', 'V', n, this->data, n, real_eigvals.get()->data(), imag_eigvals.get()->data(),
        int info = LAPACKE_dgeev( LAPACK_COL_MAJOR, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(),
                                  eigvec_0, n, eigvec_1, n );

        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm failed to compute eigenvalues.\n" );
                exit( 1 );
        }
        /* Print eigenvalues */
        //print_eigenvalues( "Eigenvalues", shape[0], wr, wi );
        print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
        /* Print left eigenvectors */
        //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
        /* Print right eigenvectors */
        //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

        std::cout << "testtest" << std::endl;
        delete eigvec_0;
        delete eigvec_1;
    }
    else{
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }

    auto return_val =  std::make_unique< DecomposeResult<double, 2, Comm<PROTOCOL::SERIAL>, ContiguousMap<2> > >( (const size_t) n,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
//    return std::make_unique< DecomposeResult<double, 2, Comm<PROTOCOL::SERIAL>, ContiguousMap<2> > >(n, 
//                                                                                                     std::move(real_eigvals),
//                                                                                                     std::move(imag_eigvals) )
}
}

