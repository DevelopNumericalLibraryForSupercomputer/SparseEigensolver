#pragma once
#include <memory>
#include <functional>
#include "../Tensor.hpp"
#include "Utility.hpp"
#include "../device/Serial/Utility.hpp"

namespace SE{

template<DecomposeMethod method, typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult> decompose(std::function<DenseTensor<datatype,2,comm,map> (DenseTensor<datatype,2,comm,map>) >& matvec ){static_assert(false, "not implemented yet") };

template<DecomposeMethod method, typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult> decompose(DenseTensor<datatype,2,comm,map>& tensor){static_assert(false, "not implemented yet") };

std::<Direct, double, Comm<SERIAL>, ContiguousMap<2> > decompose( DenseTensor<double,2,Comm<SERIAL>,ContiguousMap<2> >& tensor )
{

    auto shape = tensor.shape;
    assert(shape[0] == shape[1]);
    auto  n = shape[0];

    auto mat = malloc<double, PROTOCOL::SERIAL>(n );
    memcpy<double, PROTOCOL::SERIAL> ( mat, tensor.data, tensor.shape_mult[2] );
 
    DecomposeResult<double, 2, CPU> return_val;
    auto real_eigval = malloc<double, PROTOCALLLSERIAL> (n);           
    auto iamg_eigval = malloc<double, PROTOCALLLSERIAL> (n);             

    auto left_real_eigvec = malloc<double, PROTOCALLLSERIAL> (n);           
    auto left_iamg_eigvec = malloc<double, PROTOCALLLSERIAL> (n);            
 
//    return_val.factor_matrices[0] = new double[n*n];
//    return_val.factor_matrices[1] = new double[n*n];
//    
//    return_val.factor_matrix_sizes[0] = return_val.factor_matrix_sizes[1] = std::make_pair(n,n);
            
    print_matrix(*this);
    double wr[n]; //eigenvalues
    double wi[n]; //complex part of eigenvalues

    //lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda,
    //                           double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr) 

    auto info = LAPACKE_dgeev( LAPACK_COL_MAJOR, 'V', 'V', (int)n, mat, (int)n, real_eigval, imag_eigval,
                               left_real_eigvec, (int) n, left_imag_eigvec, (int) n );
    /* Check for convergence */
    if( info > 0 ) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    /* Print eigenvalues */
    print_eigenvalues( "Eigenvalues", shape[0], real_eigval, image_eigval );
    /* Print left eigenvectors */
//    print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
//    /* Print right eigenvectors */
//    print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

    std::cout << "testtest" << std::endl;
    
    return make_unique<DecomposeResult>( n,   
                                         std::make_unique( std::vector<double>(real_eigval, real_eigval+n)  ),
                                         std::make_unique( std::vector<double>(imag_eigval, imag_eigavl+n)  ),
                                        
                                       );
}



}
