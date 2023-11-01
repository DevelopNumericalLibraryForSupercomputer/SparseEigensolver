#pragma once
#include "../DenseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
#include <memory>

#include "DecomposeOption.hpp"
#include "../device/MKL/Utility.hpp"
#include "SparseTensor.hpp"

namespace SE{

void orthonormalize(double* eigvec, int vector_size, int number_of_vectors, std::string method){
    if(method == "qr"){
        //int LAPACKE_dgeqrf(int matrix_layout, int m, int n, double *a, int lda, double *tau)
        double* tau = malloc<double, computEnv::MKL>(number_of_vectors);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, vector_size, number_of_vectors, eigvec, vector_size, tau);
        //int LAPACKE_dorgqr(int matrix_layout, int m, int n, int k, double *a, int lda, const double *tau)
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(-1);
    }
}

int davidson_dense_worker(size_t n, double* matrix, double* eigval, double* eigvec, DecomposeOption* option){
    //imag_eigval = malloc<double, computEnv::MKL>(option->num_eigenvalues);
    //imag_eigvec = malloc<double, computEnv::MKL>(n*option->num_eigenvalues);

    int block_size = option->num_eigenvalues;
    double* guess = malloc<double, computEnv::MKL>(n*n);
    // initialization of gusss vector(s), V
    // guess : unit vector
    for(int i=0;i<option->num_eigenvalues;i++){
        //imag_eigval[i] = 0.0;
        for(int j=0;j<n;j++){
            guess[n*i+j] = 0.0;
            //imag_eigvec[n*i+j] = 0.0;
        }
        guess[n*i+i] = 1.0;
    }
    double* old_residual = malloc<double, computEnv::MKL>(n*option->num_eigenvalues);
    for(int i=0;i<n*option->num_eigenvalues;i++){
        old_residual[i] = 0.0;
    }
    for(int iter=0;iter<option->max_iterations;iter++){
        orthonormalize(guess, n, block_size, "qr");
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        double* W_iter = malloc<double, computEnv::MKL>(n*block_size);
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, n, 1.0, matrix, n, guess, n, 0.0, W_iter, n);
        
        //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
        double* rayleigh = malloc<double, computEnv::MKL>(block_size*block_size);
        gemm<double, computEnv::MKL>(ColMajor, Trans, NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
        //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
        double* rayleigh_eigval_0 = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigval_imag = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, computEnv::MKL>(block_size * block_size);
        double* rayleigh_eigvec_left = malloc<double, computEnv::MKL>(block_size * block_size);
        geev<double, computEnv::MKL>(ColMajor, 'N', 'V', block_size, rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigval_imag, rayleigh_eigvec_left, block_size, rayleigh_eigvec_0, block_size);
        
        eigenvec_sort<double, computEnv::MKL>(rayleigh_eigval_0, rayleigh_eigvec_0, block_size, block_size);
        /*
        std::cout << "eigvec" << std::endl;
        for(int index = 0;index<block_size;index++){
            
            for(int j = 0;j<option->num_eigenvalues;j++){
                std::cout << std::fixed << std::setprecision(5)  << rayleigh_eigvec_0[index*block_size + j] << " ";
            }
            std::cout << std::endl;
        }
        */
        //Ritz vector calculation, x_ki = V_k y_ki
        double* ritz_vec = malloc<double, computEnv::MKL>(n*block_size);
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, guess, n, rayleigh_eigvec_0, block_size, 0.0, ritz_vec, n); 
        
        //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
        double* residual = malloc<double, computEnv::MKL>(n * block_size);
        //lambda_ki x_ki
        for(int index = 0; index < n*block_size; index++){
            residual[index] = ritz_vec[index] * rayleigh_eigval_0[index/n];
        }
        //W_iterk y_ki - lambda_ki x_ki
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, W_iter, n, rayleigh_eigvec_0, block_size, -1.0, residual, n);
        //convergence check
        double sum_of_norm_square = 0.0;
        for(int index = 0; index < n*option->num_eigenvalues; index++){
            sum_of_norm_square += (residual[index] - old_residual[index])*(residual[index] - old_residual[index]);
        }
        /*
        std::cout << "==sum_of_norm_square==" << std::endl;
        for(int i=0;i<option->num_eigenvalues;i++){
            std::cout << std::fixed << std::setprecision(5) << i << "th     ritz_vec : ";
            for(int j=0;j<n;j++){
                std::cout << ritz_vec[i*n+j] << " ";
            }
            std::cout << "\n" << i << "th     residual : ";
            for(int j=0;j<n;j++){
                std::cout << residual[i*n+j] << " ";
            }
            std::cout << "\n" << i << "th old residual : ";
            for(int j=0;j<n;j++){
                std::cout << old_residual[i*n+j] << " ";
            }
            std::cout <<std::endl;
        }
        
        std::cout << "sum_of_norm_square : " << sum_of_norm_square << "::";
        for(int index = 0;index<block_size;index++){
            std::cout << rayleigh_eigval_0[index] << " ";
        }
        std::cout  << std::endl;
        
        std::cout << "sum_of_norm_square : " << sum_of_norm_square << std::endl;
        */
        free<double, computEnv::MKL>(W_iter);
        free<double, computEnv::MKL>(rayleigh);
        free<double, computEnv::MKL>(rayleigh_eigval_imag);
        free<double, computEnv::MKL>(rayleigh_eigvec_0);
        free<double, computEnv::MKL>(rayleigh_eigvec_left);
        if(iter != 0 && sum_of_norm_square < option->tolerance*option->tolerance){
            memcpy<double, computEnv::MKL>(eigvec, ritz_vec, n*option->num_eigenvalues);
            std::cout << rayleigh_eigval_0[0] << " ";
            memcpy<double, computEnv::MKL>(eigval, rayleigh_eigval_0, option->num_eigenvalues);
            std::cout << eigval[0] << std::endl;
            free<double, computEnv::MKL>(guess);
            free<double, computEnv::MKL>(old_residual);
            free<double, computEnv::MKL>(rayleigh_eigval_0);
            free<double, computEnv::MKL>(ritz_vec);
            free<double, computEnv::MKL>(residual);
            break;
        }
        //correction vector

        //Using diagonal preconditioner
        if(block_size > n-option->num_eigenvalues){
            return 1;
        }
        memcpy<double, computEnv::MKL>(old_residual, residual, n*option->num_eigenvalues); 
        for(int i=0;i<option->num_eigenvalues;i++){
            double coeff_i = rayleigh_eigval_0[i] - matrix[i + n*i];
            if(coeff_i > option->preconditioner_tolerance){
                for(int j=0;j<n;j++){
                    guess[n*(block_size+i) + j] = residual[n*i + j] / coeff_i;
                }
            }
        }

        block_size += option->num_eigenvalues;
        
        free<double, computEnv::MKL>(rayleigh_eigval_0);
        free<double, computEnv::MKL>(ritz_vec);
        free<double, computEnv::MKL>(residual);
    }
    return 0;
}

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::davidson(){
    DecomposeOption option;
    std::unique_ptr<double[]> real_eigvals(new double[option.num_eigenvalues]);
    std::unique_ptr<double[]> imag_eigvals(new double[option.num_eigenvalues]);

    assert(shape[0] == shape[1]);
    auto eigvec_0 = new double[option.num_eigenvalues*shape[0]];
    int info = davidson_dense_worker(shape[0], this->data, real_eigvals.get(), eigvec_0, &option);
    if( info > 0 ) {
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
    }
    for(int i=0;i<option.num_eigenvalues;i++){
        imag_eigvals.get()[i] = 0;
    }
    delete eigvec_0;
    std::cout << real_eigvals.get()[0] <<  std::endl;
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
}

void spmv(size_t n, std::vector<std::pair<std::array<size_t, 2>, double> > matrix, double* vectors, size_t number_of_vectors, double* output){
    for(int i=0;i<n;i++){
        for(int vector_index = 0; vector_index < number_of_vectors ; vector_index++){
            output[i+vector_index*n] = 0;
        }
    }
    for(auto entity : matrix){
        for(int vector_index = 0; vector_index < number_of_vectors ; vector_index++){
            output[entity.first[0] + vector_index*n] += entity.second * vectors[entity.first[1] + vector_index*n];
        }
    }
}

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
        SparseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::davidson(){
    DecomposeOption option;
    std::unique_ptr<double[]> real_eigvals(new double[option.num_eigenvalues]);
    std::unique_ptr<double[]> imag_eigvals(new double[option.num_eigenvalues]);

    assert(shape[0] == shape[1]);
    const int n = shape[0];
    auto eigvec_0 = new double[option.num_eigenvalues*n];
    
    //davidson start
    int block_size = option.num_eigenvalues;
    double* guess = malloc<double, computEnv::MKL>(n*option.max_iterations);
    // initialization of gusss vector(s), V
    // guess : unit vector
    for(int i=0;i<option.num_eigenvalues;i++){
        for(int j=0;j<n;j++){
            guess[n*i+j] = 0.0;
        }
        guess[n*i+i] = 1.0;
    }
    double* old_residual = malloc<double, computEnv::MKL>(n*option.num_eigenvalues);
    for(int i=0;i<n*option.num_eigenvalues;i++){
        old_residual[i] = 0.0;
    }
    for(int iter=0;iter<option.max_iterations;iter++){
        orthonormalize(guess, n, block_size, "qr");
        double* W_iter = malloc<double, computEnv::MKL>(n*block_size);
        spmv(n, this->data, guess, block_size, W_iter);
        
        double* rayleigh = malloc<double, computEnv::MKL>(block_size*block_size);
        gemm<double, computEnv::MKL>(ColMajor, Trans, NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
        //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
        double* rayleigh_eigval_0 = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigval_imag = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, computEnv::MKL>(block_size * block_size);
        double* rayleigh_eigvec_left = malloc<double, computEnv::MKL>(block_size * block_size);
        geev<double, computEnv::MKL>(ColMajor, 'N', 'V', block_size, rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigval_imag, rayleigh_eigvec_left, block_size, rayleigh_eigvec_0, block_size);
        
        eigenvec_sort<double, computEnv::MKL>(rayleigh_eigval_0, rayleigh_eigvec_0, block_size, block_size);
        /*
        std::cout << "eigvec" << std::endl;
        for(int index = 0;index<block_size;index++){
            
            for(int j = 0;j<option.num_eigenvalues;j++){
                std::cout <<std::fixed << std::setprecision(5) << rayleigh_eigvec_0[index*block_size + j] << " ";
            }
            std::cout << std::endl;
        }
        */
        double* ritz_vec = malloc<double, computEnv::MKL>(n*block_size);
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, guess, n, rayleigh_eigvec_0, block_size, 0.0, ritz_vec, n); 
        
        double* residual = malloc<double, computEnv::MKL>(n * block_size);
        for(int index = 0; index < n*block_size; index++){
            residual[index] = ritz_vec[index] * rayleigh_eigval_0[index/n];
        }
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, W_iter, n, rayleigh_eigvec_0, block_size, -1.0, residual, n);
        double sum_of_norm_square = 0.0;
        for(int index = 0; index < n*option.num_eigenvalues; index++){
            sum_of_norm_square += (residual[index] - old_residual[index])*(residual[index] - old_residual[index]);
        }
        /*
        std::cout << "==sum_of_norm_square==" << std::endl;
        for(int i=0;i<option.num_eigenvalues;i++){
            std::cout << std::fixed << std::setprecision(5) << i << "th     ritz_vec : ";
            for(int j=0;j<n;j++){
                std::cout << ritz_vec[i*n+j] << " ";
            }
            std::cout << "\n" << i << "th     residual : ";
            for(int j=0;j<n;j++){
                std::cout << residual[i*n+j] << " ";
            }
            std::cout << "\n" << i << "th old residual : ";
            for(int j=0;j<n;j++){
                std::cout << old_residual[i*n+j] << " ";
            }
            std::cout <<std::endl;
        }
        
        std::cout << "sum_of_norm_square : " << sum_of_norm_square << "::";
        for(int index = 0;index<block_size;index++){
            std::cout << rayleigh_eigval_0[index] << " ";
        }
        std::cout  << std::endl;
        
        std::cout << "sum_of_norm_square : " << sum_of_norm_square << std::endl;
        */
        free<double, computEnv::MKL>(W_iter);
        free<double, computEnv::MKL>(rayleigh);
        free<double, computEnv::MKL>(rayleigh_eigval_imag);
        free<double, computEnv::MKL>(rayleigh_eigvec_0);
        free<double, computEnv::MKL>(rayleigh_eigvec_left);
        if(iter != 0 && sum_of_norm_square < option.tolerance*option.tolerance){
            memcpy<double, computEnv::MKL>(eigvec_0, ritz_vec, n*option.num_eigenvalues);
            std::cout << rayleigh_eigval_0[0] << " ";
            memcpy<double, computEnv::MKL>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            std::cout << real_eigvals.get()[0] <<  std::endl;
            free<double, computEnv::MKL>(guess);
            free<double, computEnv::MKL>(old_residual);
            free<double, computEnv::MKL>(rayleigh_eigval_0);
            free<double, computEnv::MKL>(ritz_vec);
            free<double, computEnv::MKL>(residual);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            std::cout << "davidson diagonalization is not converged!" << std::endl;
            exit(1);
        }
        memcpy<double, computEnv::MKL>(old_residual, residual, n*option.num_eigenvalues); 
        std::array<size_t, 2> index;
        for(size_t i=0; i<option.num_eigenvalues; i++){
            index = {i,i};
            double coeff_i = rayleigh_eigval_0[i] - this->operator()(index);
            if(coeff_i > option.preconditioner_tolerance){
                for(int j=0;j<n;j++){
                    guess[n*(block_size+i) + j] = residual[n*i + j] / coeff_i;
                }
            }
        }
        block_size += option.num_eigenvalues;
        free<double, computEnv::MKL>(rayleigh_eigval_0);
        free<double, computEnv::MKL>(ritz_vec);
        free<double, computEnv::MKL>(residual);
    }
    for(int i=0;i<n;i++){
        imag_eigvals.get()[i] = 0;
    }
    delete eigvec_0;
    std::cout << real_eigvals.get()[0] <<  std::endl;
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
}

}

