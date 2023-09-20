#pragma once
#include "../DenseTensor.hpp"
namespace TH{
void print_matrix(DenseTensor<double,2,CPU,ContiguousMap<2,CPU> > M){
    for(size_t i = 0; i < M.shape[0]; ++i){
        for(size_t j = 0; j < M.shape[1]; ++j){
            //std::cout << M({i,j}) << " ";
            std::cout << M.data[i+M.shape[0]*j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

/* Auxiliary routine: printing eigenvalues */
void print_eigenvalues( char* desc, MKL_INT n, double* wr, double* wi ) {
        MKL_INT j;
        printf( "\n %s\n", desc );
   for( j = 0; j < n; j++ ) {
      if( wi[j] == (double)0.0 ) {
         printf( " %6.8f", wr[j] );
      } else {
         printf( " (%6.2f,%6.2f)", wr[j], wi[j] );
      }
   }
   printf( "\n" );
}

/* Auxiliary routine: printing eigenvectors */
void print_eigenvectors( char* desc, MKL_INT n, double* wi, double* v, MKL_INT ldv ) {
        MKL_INT i, j;
        printf( "\n %s\n", desc );
   for( i = 0; i < n; i++ ) {
      j = 0;
      while( j < n ) {
         if( wi[j] == (double)0.0 ) {
            printf( " %6.8f", v[i+j*ldv] );
            j++;
         } else {
            printf( " (%6.2f,%6.2f)", v[i+j*ldv], v[i+(j+1)*ldv] );
            printf( " (%6.2f,%6.2f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
            j += 2;
         }
      }
      printf( "\n" );
   }
}

template <>
DecomposeResult<double, 2, CPU> DenseTensor<double, 2, CPU, ContiguousMap<2,CPU> >::decompose(std::string method){
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
        int n = shape[0];
        DecomposeResult<double, 2, CPU> return_val;

        return_val.factor_matrices[0] = new double[n*n];
        return_val.factor_matrices[1] = new double[n*n];
        
        return_val.factor_matrix_sizes[0] = return_val.factor_matrix_sizes[1] = std::make_pair(n,n);
                
        print_matrix(*this);
        double wr[n]; //eigenvalues
        double wi[n]; //complex part of eigenvalues

        //lapack_int LAPACKE_dgeev (int matrix_layout, char jobvl, char jobvr, lapack_int n, double* a, lapack_int lda,
        //                           double* wr, double* wi, double* vl, lapack_int ldvl, double* vr, lapack_int ldvr) 

        int info = LAPACKE_dgeev( LAPACK_COL_MAJOR, 'V', 'V', n, this->data, n, wr, wi,
                        return_val.factor_matrices[0], n, return_val.factor_matrices[1], n );
        /* Check for convergence */
        if( info > 0 ) {
                printf( "The algorithm failed to compute eigenvalues.\n" );
                exit( 1 );
        }
        /* Print eigenvalues */
        print_eigenvalues( "Eigenvalues", shape[0], wr, wi );
        /* Print left eigenvectors */
        print_eigenvectors( "Left eigenvectors", shape[0], wi, return_val.factor_matrices[0], 3 );
        /* Print right eigenvectors */
        print_eigenvectors( "Right eigenvectors", shape[0], wi, return_val.factor_matrices[1], 3 );

        std::cout << "testtest" << std::endl;
    }
    else{
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }
}
}