#pragma once
#include <memory>
#include "DecomposeOption.hpp"

#include "Utility.hpp"
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"

#include "../device/TensorOp.hpp"

namespace SE{

template<typename DATATYPE, DEVICETYPE device, typename MAPTYPE>
void calculate_Witer(Tensor<STORETYPE::Dense, DATATYPE, 2, device, MAPTYPE>& tensor, DATATYPE* guess, size_t n, size_t block_size, DATATYPE* W_iter){
    gemm<DATATYPE, device>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, n, 1.0, tensor.data, n, guess, n, 0.0, W_iter, n);
}

template<typename DATATYPE, DEVICETYPE device>
void subspace_diagonalization(DATATYPE* rayleigh, size_t block_size, DATATYPE* sub_eigval, DATATYPE* sub_eigvec){
    DATATYPE* rayleigh_eigval_imag = malloc<DATATYPE, device>(block_size);
    DATATYPE* rayleigh_eigvec_left = malloc<DATATYPE, device>(block_size * block_size);

    //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
    geev<DATATYPE, device>(SE_layout::ColMajor, 'N', 'V', block_size, rayleigh, block_size, sub_eigval, rayleigh_eigval_imag, rayleigh_eigvec_left, block_size, sub_eigvec, block_size);
    free<DATATYPE, device>(rayleigh_eigval_imag);
    free<DATATYPE, device>(rayleigh_eigvec_left);
    eigenvec_sort<DATATYPE, device>(sub_eigval, sub_eigvec, block_size, block_size);
}

template<typename DATATYPE, DEVICETYPE device>
void calculate_ritz_vector(DATATYPE* guess, DATATYPE* sub_eigvec, size_t n, size_t block_size, DATATYPE* ritz_vec){
    //Ritz vector calculation, x_ki = V_k y_ki
    gemm<DATATYPE, device>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, guess, n, sub_eigvec, block_size, 0.0, ritz_vec, n); 
}

template<>
void calculate_residual<double, SEMkl>(double* W_iter, double* sub_eigval, double* sub_eigvec, double* ritz_vec, size_t n, size_t block_size, double* residual){
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    for(int index = 0; index < n*block_size; index++){
        residual[index] = ritz_vec[index] * sub_eigval[index/n];
    }
    //W_iterk y_ki - lambda_ki x_ki
    gemm<double, SEMkl>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, W_iter, n, sub_eigvec, block_size, -1.0, residual, n);
}

template<>
bool check_convergence<double, SEMkl>(const Comm<SEMkl>* _comm, double* residual, double* old_residual, size_t n, size_t num_eigenvalues, double tolerance){
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
    */
    std::cout << "sum_of_norm_square : " << sum_of_norm_square << std::endl;
    return sum_of_norm_square < tolerance*tolerance;
}

template <typename DATATYPE, typename MAPTYPE>
void preconditioner(Tensor<STORETYPE::Dense, DATATYPE, 2, SEMkl, MAPTYPE>& tensor, DecomposeOption option, DATATYPE* sub_eigval, DATATYPE* residual, size_t block_size, DATATYPE* guess){
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = tensor.shape[0];
        std::array<size_t, 2> index;
        //std::cout << "precond: ";
        for(size_t i=0; i<option.num_eigenvalues; i++){
            index = {i,i};
            DATATYPE coeff_i = sub_eigval[i] - tensor.data[i+n*i];
            if(coeff_i > option.preconditioner_tolerance){
                for(int j=0;j<n;j++){
                    guess[n*(block_size+i) + j] = residual[n*i + j] / coeff_i;
                    //std::cout << "( " << n*(block_size+i) + j << " : " << residual[n*i+j] / coeff_i << "), ";
                }
            }
        }
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
}

template <typename DATATYPE, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > davidson(Tensor<STORETYPE::Dense, DATATYPE, 2, device, MAPTYPE>& tensor){
    DecomposeOption option;
    std::unique_ptr<DATATYPE[]> real_eigvals(new DATATYPE[option.num_eigenvalues]);
    std::unique_ptr<DATATYPE[]> imag_eigvals(new DATATYPE[option.num_eigenvalues]);

    auto shape = tensor.map.get_global_shape();
    assert (shape[0] == shape[1]);
    const size_t n = shape[0];

    int block_size = option.num_eigenvalues;
    // initialization of gusss vector(s), V
    // guess : unit vector
    DATATYPE* guess = malloc<DATATYPE, device>(n*block_size*option.max_iterations);
    memset<DATATYPE, device>(guess, 0.0, n*block_size*option.max_iterations);

    for(int i=0;i<option.num_eigenvalues;i++){
        guess[i*n+i] = 1.0;
    }
    DATATYPE* old_residual = malloc<DATATYPE, device>(n*option.num_eigenvalues);
    memset<DATATYPE, device>(old_residual, 0.0, n*option.num_eigenvalues);
    
    int iter = 0;
// zero-th step.
    orthonormalize<DATATYPE, device>(guess, n, block_size, "qr");




    while(iter<option.max_iterations){
        orthonormalize<DATATYPE, device>(guess, n, block_size, "qr");
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        DATATYPE* W_iter = malloc<DATATYPE, device>(n*block_size);
        calculate_Witer(tensor, guess, n, block_size, W_iter);
/*
    std::cout << "guess" << std::endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << guess[i+j*n] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;     
    std::cout << "Witer" << std::endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << W_iter[i+j*n] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
*/
        DATATYPE* rayleigh = malloc<DATATYPE, device>(block_size* block_size);
        DATATYPE* rayleigh_eigval_0 = malloc<DATATYPE, device>(block_size);
        DATATYPE* rayleigh_eigvec_0 = malloc<DATATYPE, device>(block_size * block_size);
        
        //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
        gemm<DATATYPE, device>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
/*
    std::cout << "rayleigh" << std::endl;
    for(int i=0;i<block_size;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << rayleigh[i+j*block_size] << " ";
        }
        std::cout << std::endl;
    }
*/
        subspace_diagonalization<DATATYPE, device>(rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigvec_0);

/*
    std::cout << "=======================" << std::endl;
    std::cout << "diag:" << std::endl;
    for(int i=0;i<block_size;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << rayleigh_eigvec_0[i+j*block_size] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
*/

        //calculate ritz vector
        DATATYPE* ritz_vec = malloc<DATATYPE, device>(n*block_size);
        calculate_ritz_vector<DATATYPE, device>(guess, rayleigh_eigvec_0, n, block_size, ritz_vec);
/*
    std::cout << "ritz:" << std::endl;
    for(int i=0;i<n;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << ritz_vec[i+j*n] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
*/
        DATATYPE* residual = malloc<DATATYPE, device>(n*block_size);
        calculate_residual<DATATYPE, device>(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<DATATYPE, device>(W_iter);

        bool is_converged = check_convergence<DATATYPE, device>(tensor.comm, residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            //memcpy<DATATYPE, device>(eigvec_0.get(), ritz_vec, n*option.num_eigenvalues);
            memcpy<DATATYPE, device>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }
        preconditioner(tensor, option,rayleigh_eigval_0, residual, block_size, guess);
/*
         std::cout << "\nguess: ";
        for(int i=0;i<n*(block_size+1);i++){
            std::cout << std::setw(3) << guess[i] << " ";
        }
        std::cout << std::endl;
*/
        memcpy<DATATYPE, device>(old_residual, residual, n*option.num_eigenvalues);
        block_size += option.num_eigenvalues;
        iter++;
        free<DATATYPE, device>(rayleigh_eigval_0);
        free<DATATYPE, device>(rayleigh_eigvec_0);
        free<DATATYPE, device>(ritz_vec);
        free<DATATYPE, device>(residual);
    }
    if(iter == option.max_iterations){
        std::cout << "diagonalization did not converged!" << std::endl;
        exit(-1);
    }
    else{
        for(int i=0;i<option.num_eigenvalues;i++){
            imag_eigvals.get()[i] = 0;
        }
    
        std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));

        return std::move(return_val);
    }
}

}

