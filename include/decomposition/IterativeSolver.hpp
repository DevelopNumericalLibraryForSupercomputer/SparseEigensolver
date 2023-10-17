#pragma once
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
#include <memory>

#include "DecomposeOption.hpp"
#include "../device/MKL/Utility.hpp"

namespace SE{

void orthonormalize(double* eigvec, std::string method){
    DecomposeOption option;
    if(method == "qr"){
        std::cout << "not implemented" << std::endl;
    }
    else if(method == "cholesky"){
        std::cout << "not implemented" << std::endl;
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(-1);
    }
}


int davidson_real(size_t n, double* matrix, double* eigval, double* eigvec){
    DecomposeOption option;

    int num_eigenvalues = option.num_eigenvalues;
    double* guess = new double[n*n];
    double* rayliegh = new double[n*n];
    // initialization of gusss vector(s), V
    // guess : unit vector
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            guess[n*i+j] = 0.0;
        }
        if(i < num_eigenvalues){
            guess[n*i+i] = 1.0;
        }        
    }
    for(int iter=0;iter<option.max_iterations;iter++){
        //guess vector를 orthomormalize
        orthonormalize(guess, "qr");
        //subspace matrix (Rayleigh matrix) 생성
        // T = V^t A V
        // i.e. 2D matrix : column major, i + row*j
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans,
                                

        )
        //submatrix 대각화
        //residual 계산
        //correction vector 계산
        //correction, or converged

    }
    


}


template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::davidson(const std::string method){

    const int n = shape[0];
    //auto real_eigvals=std::make_unique< std::vector<double> > (n*n);
    //auto imag_eigvals=std::make_unique< std::vector<double> > (n*n);
    std::unique_ptr<double[]> real_eigvals(new double[n]);
    std::unique_ptr<double[]> imag_eigvals(new double[n]);

    if(method.compare("Davidson")==0){
        //eigenvalue decomposition, N by N matrix A = Q Lambda Q-1
        //factor matricies : Q, Q
        //factor matrix sizes : N,N / N,N
        //core tensor = Lambda matrix
        std::cout << "Davidson Start!" << std::endl;
        
        assert(shape[0] == shape[1]);

        //return_val.factor_matrices[0] = new double[n*n];
        //return_val.factor_matrices[1] = new double[n*n];
        auto eigvec_0 = new double[n*n];
        auto eigvec_1 = new double[n*n];
        
        //return_val.factor_matrix_sizes[0] = return_val.factor_matrix_sizes[1] = std::make_pair(n,n);

        //lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda,
        //                           double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr) 

        int info = davidson_real(n, this->data, real_eigvals.get(), eigvec_0);

        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm failed to compute eigenvalues.\n" );
                exit( 1 );
        }
        /* Print eigenvalues */
        //print_eigenvalues( "Eigenvalues", shape[0], real_eigvals.get(), imag_eigvals.get() );
        /* Print left eigenvectors */
        //print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
        /* Print right eigenvectors */
        //print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

        delete eigvec_0;
        delete eigvec_1;
    }
    else{
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }

    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (const size_t) n,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
//    return std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >(n, 
//                                                                                                     std::move(real_eigvals),
//                                                                                                     std::move(imag_eigvals) )
}



}

