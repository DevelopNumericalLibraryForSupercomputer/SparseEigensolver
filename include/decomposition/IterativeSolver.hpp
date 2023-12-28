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
    //double norm[num_eigenvalues];
    //bool is_converged = true;
    for(int i=0;i<num_eigenvalues;i++){
        double norm = nrm2<DATATYPE, device>(residual.map.get_global_shape()[0], &residual.data[i], residual.map.get_global_shape()[1]);
        if(norm > tolerance){
            //std::cout << "residual norm of " << i << "th vector : " << std::scientific << std::setprecision(4) << norm[i] << std::endl;
            return false;
        }
    }
    return true;
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

    // initialization of gusss vector(s), V
    std::array<size_t, 2> guess_shape = {vec_size,option.num_eigenvalues};
    auto guess_map = MAPTYPE(guess_shape, rank, world_size);
    DenseTensor<2, DATATYPE, MAPTYPE, device> guess(*tensor.copy_comm(), guess_map);
    // guess : unit vector
    for(size_t i=0;i<option.num_eigenvalues;i++){
        std::array<size_t, 2> tmp_index = {i,i};
        guess.global_set_value(tmp_index, 1.0);
    }
    // univ vector guess : do not need orthonormalize.

    bool return_result = false;
    size_t iter = 0;
    while(iter < option.max_iterations){
        //auto new_guess = std::make_unique<DenseTensor<2, DATATYPE, MAPTYPE, device> >(guess);
        DenseTensor<2, DATATYPE, MAPTYPE, device>* new_guess = new DenseTensor<2, DATATYPE, MAPTYPE, device>(guess);
        //std::cout << *new_guess << std::endl;
        //block expansion loop
        size_t i_block = 0;
        
        while(true){
            size_t block_size = option.num_eigenvalues*(i_block+1);
            // W_iterk = A V_k
            //auto w_iter = TensorOp::matmul(tensor, *new_guess.get(), TRANSTYPE::N, TRANSTYPE::N);
            //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
            //auto subspace_matrix = TensorOp::matmul(*new_guess.get(), w_iter, TRANSTYPE::T, TRANSTYPE::N);
            
            auto w_iter = TensorOp::matmul(tensor, *new_guess, TRANSTYPE::N, TRANSTYPE::N);
            
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
            //auto ritz_vec = TensorOp::matmul(*new_guess.get(), sub_eigvec, TRANSTYPE::N, TRANSTYPE::N);
            //ritz vectors are new eigenvector candidates
            
            //get residual
            auto residual = calculate_residual(w_iter, sub_eigval, sub_eigvec, ritz_vec);
            //check convergence
            bool is_converged = check_convergence(residual, option.num_eigenvalues, option.tolerance);
            if(is_converged){
                return_result = true;
                memset<DATATYPE, device>(imag_eigvals.get(), 0.0, option.num_eigenvalues);
                memcpy<DATATYPE, device>(real_eigvals.get(), sub_eigval, option.num_eigenvalues);
                break;
            }
            if(i_block == option.max_block-1){
                for(int i=0;i<option.num_eigenvalues;i++){
                    copy<DATATYPE, device>(vec_size, &ritz_vec.data[i], option.max_block*option.num_eigenvalues, &guess.data[i], option.num_eigenvalues);
                }
                break;
            }
            //preconditioning
            //"Diagonal preconditioner" return expanded guess matrix.
            DenseTensor<2, DATATYPE, MAPTYPE, device> tmp_guess = preconditioner(tensor, residual, ritz_vec, sub_eigval, option);
            free<device>(sub_eigval);
            delete new_guess;
            new_guess = new DenseTensor<2, DATATYPE, MAPTYPE, device>(tmp_guess);//std::make_unique<DenseTensor<2, DATATYPE, MAPTYPE, device> >(tmp_guess);
            
            i_block++;
        }
        delete new_guess;
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
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));
 
    return std::move(return_val);
}
}

