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
template <typename DATATYPE, typename MAPTYPE1, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE1, device> calculate_residual( // return residual (block_size by vec_size)
    DenseTensor<2, DATATYPE, MAPTYPE1, device> w_iter,      //vec_size by block_size
    DATATYPE* sub_eigval,                                  //block_size
    DenseTensor<2, DATATYPE, MAPTYPE1, device> sub_eigvec,  //block_size by block_size
    DenseTensor<2, DATATYPE, MAPTYPE1, device> ritz_vec)    //vec_size by block_size
{
    size_t block_size = sub_eigvec.map.get_global_shape()[0];
    size_t vec_size = ritz_vec.map.get_global_shape()[0];
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    DenseTensor<2, DATATYPE, MAPTYPE1, device> residual(ritz_vec);
    for(int index = 0; index < block_size; index++){
        scal<DATATYPE, device>(vec_size, sub_eigval[index], &residual.data[index], block_size);
    }
    //W_iterk y_ki - lambda_ki x_ki
    gemm<DATATYPE, device>(ORDERTYPE::ROW, TRANSTYPE::N, TRANSTYPE::N, vec_size, block_size, block_size, 1.0, w_iter.data, block_size, sub_eigvec.data, block_size, -1.0, residual.data, block_size);
    return residual;
}

template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
bool check_convergence(DenseTensor<2, DATATYPE, MAPTYPE, device> residual, size_t num_eigenvalues, double tolerance){
    //convergence check
    double sum_of_norm = 0;
    for(int i=0;i<num_eigenvalues;i++){
        sum_of_norm += nrm2<DATATYPE, device>(residual.map.get_global_shape()[0], &residual.data[i], residual.map.get_global_shape()[1]);
    }
    std::cout << "sum_of_norm : " << sum_of_norm << std::endl;
    return sum_of_norm < tolerance;
}

template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
DenseTensor<2, DATATYPE, MAPTYPE, device> preconditioner(               
                    DenseTensor<2, DATATYPE, MAPTYPE, device> tensor,   //vec_size * vec_size
                    DenseTensor<2, DATATYPE, MAPTYPE, device> residual, //vec_size * block_size
                    DenseTensor<2, DATATYPE, MAPTYPE, device> guess,    //vec_size * block_size
                                                                        //return new guess = vec_size * new_block_size
                    DATATYPE* sub_eigval,                               //block_size
                    DecomposeOption option){
    size_t vec_size = residual.map.get_global_shape()[0];
    size_t block_size = residual.map.get_global_shape()[1];
    size_t num_eig = option.num_eigenvalues;
    size_t new_block_size = block_size + num_eig;
    
    std::array<size_t, 2> new_guess_shape = {vec_size, new_block_size};
    auto new_guess_map = MAPTYPE(new_guess_shape, tensor.comm.get_rank(), tensor.comm.get_world_size());
    DenseTensor<2, DATATYPE, MAPTYPE, device> new_guess(*tensor.copy_comm(), new_guess_map);
    //copy old guesses
    for(int i=0;i<block_size;i++){
        copy<DATATYPE, device>(vec_size, &guess.data[i], block_size, &new_guess.data[i], new_block_size);
    }

    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        std::array<size_t, 2> array_index;
        for(size_t index=0; index<option.num_eigenvalues; index++){
            array_index = {index, index};
            DATATYPE coeff_i = sub_eigval[index] - tensor(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
            if(abs(coeff_i) > option.preconditioner_tolerance){
                copy<DATATYPE, device>(vec_size, &residual.data[index], block_size, &new_guess.data[block_size+index], new_block_size);
                scal<DATATYPE, device>(vec_size, 1.0/coeff_i, &new_guess.data[block_size + index], new_block_size);
            }
        }
        TensorOp::orthonormalize(new_guess, "qr");
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
    return new_guess;
}

template <typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
std::unique_ptr<DecomposeResult<DATATYPE> > davidson(DenseTensor<2, DATATYPE, MAPTYPE, device>& tensor){
    DecomposeOption option;
    
    std::unique_ptr<DATATYPE[]> real_eigvals(new DATATYPE[option.num_eigenvalues]);
    std::unique_ptr<DATATYPE[]> imag_eigvals(new DATATYPE[option.num_eigenvalues]);

    auto shape = tensor.map.get_global_shape();
    assert (shape[0] == shape[1]);
    const size_t vec_size = shape[0];

    auto current_comm = tensor.copy_comm();
    auto rank = tensor.comm.get_rank();
    auto world_size = tensor.comm.get_world_size();

    // zero-th step.
    // initialization of gusss vector(s), V
    std::array<size_t, 2> guess_shape = {vec_size,option.num_eigenvalues};
    auto guess_map = MAPTYPE(guess_shape, rank, world_size);
    DenseTensor<2, DATATYPE, MAPTYPE, device> guess(*tensor.copy_comm(), guess_map);
    // guess : unit vector
    for(size_t i=0;i<option.num_eigenvalues;i++){
        std::array<size_t, 2> tmp_index = {i,i};
        guess.global_set_value(tmp_index, 1.0);
    }
    // 0th iteration do not need orthonormalize (up to now).
    //orthonormalize<DATATYPE, device>(guess, n, block_size, "qr");

    bool return_result = false;
    size_t iter = 0;
    while(iter < option.max_iterations){
        DenseTensor<2, DATATYPE, MAPTYPE, device>* new_guess;
        new_guess = &guess;
        //block expansion loop
        size_t i_block = 0;
        
        while(true){
            if(i_block == option.max_block){
                for(int i=0;i<option.num_eigenvalues;i++){
                    copy<DATATYPE, device>(vec_size, &new_guess->data[i], option.max_block*option.num_eigenvalues, &guess.data[i], option.num_eigenvalues);
                }
                break;
            }

            std::cout << "iter : " << iter << " , iblock : " << i_block << " / " << option.max_block << std::endl;

            size_t block_size = option.num_eigenvalues*(i_block+1);
            // W_iterk = A V_k
            auto w_iter = TensorOp::matmul(tensor, *new_guess, TRANSTYPE::N, TRANSTYPE::N);

            //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
            auto subspace_matrix = TensorOp::matmul(*new_guess, w_iter, TRANSTYPE::T, TRANSTYPE::N);
        
            //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
            DATATYPE* sub_eigval = malloc<DATATYPE, device>(block_size);
            DenseTensor<2, DATATYPE, MAPTYPE, device> sub_eigvec(subspace_matrix);
            int info = syev<DATATYPE, device>(ORDERTYPE::ROW, 'V', 'U', block_size, sub_eigvec.data, block_size, sub_eigval);
            if(info !=0){
                std::cout << "subspace_diagonalization error!" << std::endl;
                exit(-1);
            }
    
            //calculate ritz vector
            //Ritz vector calculation, x_ki = V_k y_ki
            auto ritz_vec = TensorOp::matmul(*new_guess, sub_eigvec, TRANSTYPE::N, TRANSTYPE::N);
            //ritz vectors are new eigenvector candidates
    
            std::cout << ritz_vec << std::endl;
            for(int i=0;i<option.num_eigenvalues;i++) {std::cout << sub_eigval[i] << ' ';}
            std::cout << std::endl;

            //get residual
            auto residual = calculate_residual(w_iter, sub_eigval, sub_eigvec, ritz_vec);
            //std::cout << "0-th step, residual\n" << residual << std::endl;
            //check convergence
            bool is_converged = check_convergence(residual, option.num_eigenvalues, option.tolerance);
            if(is_converged){
                return_result = true;
                memset<DATATYPE, device>(imag_eigvals.get(), 0.0, option.num_eigenvalues);
                memcpy<DATATYPE, device>(real_eigvals.get(), sub_eigval, option.num_eigenvalues);
                break;
            }
            //preconditioning
            //"Diagonal preconditioner" return expanded guess matrix.
            auto tmp_guess = preconditioner(tensor, residual, ritz_vec, sub_eigval, option);
            new_guess = &tmp_guess;
            
            i_block++;
        }
        if(return_result){                
            break;
        }
        else{
            iter++;
        }
    }
    if(!return_result){
        std::cout << "NOT CONVERGED!" << std::endl;
        exit(-1);
    }


//         orthonormalize<DATATYPE, device>(guess, n, block_size, "qr");
         // W_iterk = A V_k
         // W_iter should be updated everytime because of the numerical instability
// /*
//     std::cout << "guess" << std::endl;
//     for(int i=0;i<n;i++){
//         for(int j=0;j<block_size;j++){
//             std::cout << std::setw(6) << guess[i+j*n] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======================" << std::endl;     
//     std::cout << "Witer" << std::endl;
//     for(int i=0;i<n;i++){
//         for(int j=0;j<block_size;j++){
//             std::cout << std::setw(6) << W_iter[i+j*n] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======================" << std::endl;
// */
//         DATATYPE* rayleigh = malloc<DATATYPE, device>(block_size* block_size);
//         DATATYPE* rayleigh_eigval_0 = malloc<DATATYPE, device>(block_size);
//         DATATYPE* rayleigh_eigvec_0 = malloc<DATATYPE, device>(block_size * block_size);
//         
//         //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
//         gemm<DATATYPE, device>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
// /*
//     std::cout << "rayleigh" << std::endl;
//     for(int i=0;i<block_size;i++){
//         for(int j=0;j<block_size;j++){
//             std::cout << std::setw(6) << rayleigh[i+j*block_size] << " ";
//         }
//         std::cout << std::endl;
//     }
// */
//         subspace_diagonalization<DATATYPE, device>(rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigvec_0);
// 
// /*
//     std::cout << "=======================" << std::endl;
//     std::cout << "diag:" << std::endl;
//     for(int i=0;i<block_size;i++){
//         for(int j=0;j<block_size;j++){
//             std::cout << std::setw(6) << rayleigh_eigvec_0[i+j*block_size] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======================" << std::endl;
// */
// 
//         //calculate ritz vector
//         DATATYPE* ritz_vec = malloc<DATATYPE, device>(n*block_size);
//         calculate_ritz_vector<DATATYPE, device>(guess, rayleigh_eigvec_0, n, block_size, ritz_vec);
// /*
//     std::cout << "ritz:" << std::endl;
//     for(int i=0;i<n;i++){
//         for(int j=0;j<block_size;j++){
//             std::cout << std::setw(6) << ritz_vec[i+j*n] << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << "=======================" << std::endl;
// */
//         DATATYPE* residual = malloc<DATATYPE, device>(n*block_size);
//         calculate_residual<DATATYPE, device>(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
//         free<DATATYPE, device>(W_iter);
// 
//         bool is_converged = check_convergence<DATATYPE, device>(tensor.comm, residual, old_residual, n, option.num_eigenvalues, option.tolerance);
//         if(iter != 0 && is_converged){
//             //memcpy<DATATYPE, device>(eigvec_0.get(), ritz_vec, n*option.num_eigenvalues);
//             memcpy<DATATYPE, device>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
//             break;
//         }
//         //correction vector
//         //Using diagonal preconditioner
//         if(block_size > n-option.num_eigenvalues){
//             printf( "The algorithm failed to compute eigenvalues.\n" );
//             exit( 1 );
//         }
//         preconditioner(tensor, option,rayleigh_eigval_0, residual, block_size, guess);
// /*
//          std::cout << "\nguess: ";
//         for(int i=0;i<n*(block_size+1);i++){
//             std::cout << std::setw(3) << guess[i] << " ";
//         }
//         std::cout << std::endl;
// */
//         memcpy<DATATYPE, device>(old_residual, residual, n*option.num_eigenvalues);
//         block_size += option.num_eigenvalues;
//         iter++;
//         free<DATATYPE, device>(rayleigh_eigval_0);
//         free<DATATYPE, device>(rayleigh_eigvec_0);
//         free<DATATYPE, device>(ritz_vec);
//         free<DATATYPE, device>(residual);
//     }
//     if(iter == option.max_iterations){
//         std::cout << "diagonalization did not converged!" << std::endl;
//         exit(-1);
//     }
//     else{
//         for(int i=0;i<option.num_eigenvalues;i++){
//             imag_eigvals.get()[i] = 0;
//         }
//     
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));
 
    return std::move(return_val);
}
}

