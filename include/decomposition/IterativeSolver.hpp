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

void orthonormalize(double* eigvec, int vector_size, int number_of_vectors, std::string method){
    if(method == "qr"){
        double* tau = malloc<double, computEnv::MKL>(number_of_vectors);
        LAPACKE_dgeqrf(LAPACK_COL_MAJOR, vector_size, number_of_vectors, eigvec, vector_size, tau);
        LAPACKE_dorgqr(LAPACK_COL_MAJOR, vector_size, number_of_vectors, number_of_vectors, eigvec, vector_size, tau);
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(-1);
    }
}

template<typename datatype, typename comm>
void get_rayleigh(datatype* guess, datatype* W_iter, size_t n, size_t block_size, datatype* rayleigh){
    std::cout << "not implemented" << std::endl;
}
template<>
void get_rayleigh<double, Comm<computEnv::MKL> >(double* guess, double* W_iter, size_t n, size_t block_size, double* rayleigh){
    //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
    gemm<double, computEnv::MKL>(ColMajor, Trans, NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
}

template<typename datatype, typename comm>
void subspace_diagonalization(datatype* guess, datatype* rayleigh, size_t n, size_t block_size, datatype* sub_eigval, datatype* sub_eigvec, datatype* ritz_vec){
    std::cout << "not implemented" << std::endl;
}
template<>
void subspace_diagonalization<double, Comm<computEnv::MKL> >(double* guess, double* rayleigh, size_t n, size_t block_size, double* sub_eigval, double* sub_eigvec, double* ritz_vec){
    double* rayleigh_eigval_imag = malloc<double, computEnv::MKL>(block_size);
    double* rayleigh_eigvec_left = malloc<double, computEnv::MKL>(block_size * block_size);

    //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
    geev<double, computEnv::MKL>(ColMajor, 'N', 'V', block_size, rayleigh, block_size, sub_eigval, rayleigh_eigval_imag, rayleigh_eigvec_left, block_size, sub_eigvec, block_size);
    eigenvec_sort<double, computEnv::MKL>(sub_eigval, sub_eigvec, block_size, block_size);
    //Ritz vector calculation, x_ki = V_k y_ki
    gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, guess, n, sub_eigvec, block_size, 0.0, ritz_vec, n); 
}

template<typename datatype, typename comm>
void calculate_residual(datatype* W_iter, datatype* sub_eigval, datatype* sub_eigvec, datatype* ritz_vec, size_t n, size_t block_size, datatype* residual){
    std::cout << "not implemented" << std::endl;
}
template<>
void calculate_residual<double, Comm<computEnv::MKL> >(double* W_iter, double* sub_eigval, double* sub_eigvec, double* ritz_vec, size_t n, size_t block_size, double* residual){
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    for(int index = 0; index < n*block_size; index++){
        residual[index] = ritz_vec[index] * sub_eigval[index/n];
    }
    //W_iterk y_ki - lambda_ki x_ki
    gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, block_size, 1.0, W_iter, n, sub_eigvec, block_size, -1.0, residual, n);
}

template<typename datatype, typename comm>
bool check_convergence(datatype* residual, datatype* old_residual, size_t n, size_t num_eigenvalues, datatype tolerance){
    std::cout << "not implemented" << std::endl;
    exit(1);
}
template<>
bool check_convergence<double, Comm<computEnv::MKL> >(double* residual, double* old_residual, size_t n, size_t num_eigenvalues, double tolerance){
    //convergence check
    double sum_of_norm_square = 0.0;
    for(int index = 0; index < n*num_eigenvalues; index++){
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
    return sum_of_norm_square < tolerance*tolerance;
}

template <>
void DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess){
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = this->shape[0];
        for(size_t i=0; i<option.num_eigenvalues; i++){
            double coeff_i = sub_eigval[i] - this->data[i + n*i];
            if(coeff_i > option.preconditioner_tolerance){
                for(int j=0;j<n;j++){
                    guess[n*(block_size+i) + j] = residual[n*i + j] / coeff_i;
                }
            }
        }
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
}
template <>
void SparseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess){
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = this->shape[0];
        std::array<size_t, 2> index;
        for(size_t i=0; i<option.num_eigenvalues; i++){
            index = {i,i};
            double coeff_i = sub_eigval[i] - this->operator()(index);
            if(coeff_i > option.preconditioner_tolerance){
                for(int j=0;j<n;j++){
                    guess[n*(block_size+i) + j] = residual[n*i + j] / coeff_i;
                }
            }
        }
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
}

template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > > 
        DenseTensor<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> >::davidson(){
    DecomposeOption option;
    std::unique_ptr<double[]> real_eigvals(new double[option.num_eigenvalues]);
    std::unique_ptr<double[]> imag_eigvals(new double[option.num_eigenvalues]);

    assert(shape[0] == shape[1]);
    const size_t n = shape[0];

    auto eigvec_0 = new double[option.num_eigenvalues*shape[0]];

    int block_size = option.num_eigenvalues;
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }

    double* guess = malloc<double, computEnv::MKL>(n*block_size*option.max_iterations);
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
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        double* W_iter = malloc<double, computEnv::MKL>(n*block_size);
        gemm<double, computEnv::MKL>(ColMajor, NoTrans, NoTrans, n, block_size, n, 1.0, this->data, n, guess, n, 0.0, W_iter, n);
        
        double* rayleigh = malloc<double, computEnv::MKL>(block_size* block_size);
        double* rayleigh_eigval_0 = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, computEnv::MKL>(block_size * block_size);
        double* ritz_vec = malloc<double, computEnv::MKL>(n*block_size);
        double* residual = malloc<double, computEnv::MKL>(n*block_size);

        get_rayleigh<double, Comm<computEnv::MKL> >(guess, W_iter, n, block_size, rayleigh);
        subspace_diagonalization<double, Comm<computEnv::MKL> >(guess, rayleigh, n, block_size, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec);
        calculate_residual<double, Comm<computEnv::MKL> >(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<double, computEnv::MKL>(W_iter);

        double is_converged = check_convergence<double, Comm<computEnv::MKL> >(residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            memcpy<double, computEnv::MKL>(eigvec_0, ritz_vec, n*option.num_eigenvalues);
            memcpy<double, computEnv::MKL>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }
        preconditioner(option,rayleigh_eigval_0, residual, block_size, guess);

        block_size += option.num_eigenvalues;
        
        free<double, computEnv::MKL>(rayleigh_eigval_0);
        free<double, computEnv::MKL>(rayleigh_eigvec_0);
        free<double, computEnv::MKL>(ritz_vec);
        free<double, computEnv::MKL>(residual);
    }

    for(int i=0;i<option.num_eigenvalues;i++){
        imag_eigvals.get()[i] = 0;
    }
    delete eigvec_0;
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
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
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }
    double* guess = malloc<double, computEnv::MKL>(n*block_size*option.max_iterations);
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
        
        double* rayleigh = malloc<double, computEnv::MKL>(block_size* block_size);
        double* rayleigh_eigval_0 = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, computEnv::MKL>(block_size * block_size);
        double* ritz_vec = malloc<double, computEnv::MKL>(n*block_size);
        double* residual = malloc<double, computEnv::MKL>(n * block_size);

        get_rayleigh<double, Comm<computEnv::MKL> >(guess, W_iter, n, block_size, rayleigh);
        subspace_diagonalization<double, Comm<computEnv::MKL> >(guess, rayleigh, n, block_size, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec);
        calculate_residual<double, Comm<computEnv::MKL> >(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<double, computEnv::MKL>(W_iter);

        double is_converged = check_convergence<double, Comm<computEnv::MKL> >(residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            memcpy<double, computEnv::MKL>(eigvec_0, ritz_vec, n*option.num_eigenvalues);
            memcpy<double, computEnv::MKL>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            std::cout << "davidson diagonalization is not converged!" << std::endl;
            exit(1);
        }
        preconditioner(option, rayleigh_eigval_0, residual, block_size, guess);
        block_size += option.num_eigenvalues;
        free<double, computEnv::MKL>(rayleigh);
        free<double, computEnv::MKL>(rayleigh_eigval_0);
        free<double, computEnv::MKL>(ritz_vec);
        free<double, computEnv::MKL>(residual);
    }
    for(int i=0;i<option.num_eigenvalues;i++){
        imag_eigvals.get()[i] = 0;
    }
    free<double,computEnv::MKL>(eigvec_0);
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MKL>, ContiguousMap<2> > >( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals));

    return std::move(return_val);
}

/*
template <>
std::unique_ptr<DecomposeResult<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> > > 
        SparseTensor<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> >::davidson(){
    DecomposeOption option;
    std::unique_ptr<double[]> real_eigvals(new double[option.num_eigenvalues]);
    std::unique_ptr<double[]> imag_eigvals(new double[option.num_eigenvalues]);

    assert(shape[0] == shape[1]);
    const size_t n = shape[0];


    std::array<size_t, 2> eigvec_shape = {n, option.num_eigenvalues};
    auto eigvec_map = ContiguousMap<2>(eigvec_shape);

    auto eigvec_0 = new DenseTensor<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> >(eigvec_shape);
    
    //davidson start
    int block_size = option.num_eigenvalues;
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }

    DenseTensor<double, 2, Comm<computeEnv::MPI>, contiguousMap<2> > guess = new DenseTensor<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> >({n, block_size*option.max_iterations});
    
    // initialization of gusss vector(s), V
    // guess : unit vector
    
    for(int i=0;i<option.num_eigenvalues;i++){
        for(int j=0;j<n;j++){
            guess[n*i+j] = 0.0;
        }
        guess[n*i+i] = 1.0;
    }
    
    double* old_residual = malloc<double, computEnv::MPI>(n*option.num_eigenvalues);
    for(int i=0;i<n*option.num_eigenvalues;i++){
        old_residual[i] = 0.0;
    }
    std::cout << "not completed" << std::endl;
    
    for(int iter=0;iter<option.max_iterations;iter++){
        orthonormalize(guess, n, block_size, "qr");

        double* W_iter = malloc<double, computEnv::MKL>(n*block_size);
        spmv(n, this->data, guess, block_size, W_iter);
        
        double* rayleigh = malloc<double, computEnv::MKL>(block_size* block_size);
        double* rayleigh_eigval_0 = malloc<double, computEnv::MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, computEnv::MKL>(block_size * block_size);
        double* ritz_vec = malloc<double, computEnv::MKL>(n*block_size);
        double* residual = malloc<double, computEnv::MKL>(n * block_size);

        get_rayleigh<double, Comm<computEnv::MKL> >(guess, W_iter, n, block_size, rayleigh);
        subspace_diagonalization<double, Comm<computEnv::MKL> >(guess, rayleigh, n, block_size, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec);
        calculate_residual<double, Comm<computEnv::MKL> >(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<double, computEnv::MKL>(W_iter);

        double is_converged = check_convergence<double, Comm<computEnv::MKL> >(residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            memcpy<double, computEnv::MKL>(eigvec_0, ritz_vec, n*option.num_eigenvalues);
            memcpy<double, computEnv::MKL>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            std::cout << "davidson diagonalization is not converged!" << std::endl;
            exit(1);
        }
        preconditioner(option, rayleigh_eigval_0, residual, block_size, guess);
        block_size += option.num_eigenvalues;
        free<double, computEnv::MKL>(rayleigh);
        free<double, computEnv::MKL>(rayleigh_eigval_0);
        free<double, computEnv::MKL>(ritz_vec);
        free<double, computEnv::MKL>(residual);
    }
    for(int i=0;i<option.num_eigenvalues;i++){
        imag_eigvals.get()[i] = 0;
    }
    free<double,computEnv::MKL>(eigvec_0);
    
    for(int i=0;i<option.num_eigenvalues;i++){
        real_eigvals.get()[i] = -1;
        imag_eigvals.get()[i] = 0;
    }
    auto return_val = std::make_unique< DecomposeResult<double, 2, Comm<computEnv::MPI>, ContiguousMap<2> > >( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals));
    
    return std::move(return_val);
}

*/

}

