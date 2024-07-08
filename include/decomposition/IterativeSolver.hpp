#pragma once
#include <memory>
#include "DecomposeOption.hpp"
#include "../Utility.hpp"
#include "../device/TensorOp.hpp"
#include "TensorOperations.hpp"
#include "DecomposeResult.hpp"

namespace SE{
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device> calculate_residual( // return residual (block_size by vec_size)
    DenseTensor<2, DATATYPE, mtype, device>& w_iter,      //vec_size by block_size
    DATATYPE* sub_eigval,                                   //block_size
    DenseTensor<2, DATATYPE, mtype, device>& sub_eigvec,  //block_size by block_size
    DenseTensor<2, DATATYPE, mtype, device>& ritz_vec)    //vec_size by block_size
{
    //int block_size = sub_eigvec.map.get_global_shape()[0];
    //int vec_size = ritz_vec.map.get_global_shape()[0];
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    DenseTensor<2, DATATYPE, mtype, device> scaled_ritz(ritz_vec);
    TensorOp::scale_vectors(scaled_ritz, sub_eigval);
    //W_iterk y_ki - lambda_ki x_ki
    auto tmp_residual = TensorOp::matmul(w_iter, sub_eigvec);
    auto residual = TensorOp::add(tmp_residual, scaled_ritz, -1.0);
    return residual;
}

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
bool check_convergence(const DenseTensor<2, DATATYPE, mtype, device>& residual, const int num_eigenvalues, const double tolerance){
    //convergence check
    double* norm = malloc<DATATYPE, device>(num_eigenvalues);
    TensorOp::get_norm_of_vectors(residual, norm, num_eigenvalues);
    for(int i=0;i<num_eigenvalues;i++){
        if(norm[i] > tolerance){
            free<device>(norm);
            return false;
        }
    }
    free<device>(norm);
    return true;
}

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
DenseTensor<2, DATATYPE, mtype, device>  preconditioner(               
//                    DenseTensor<2, DATATYPE, mtype, device> tensor,   //vec_size * vec_size
                    const TensorOperations<mtype, device>* operations,   //vec_size * vec_size
                    DenseTensor<2, DATATYPE, mtype, device>& residual, //vec_size * block_size
                    DenseTensor<2, DATATYPE, mtype, device>& guess,    //vec_size * block_size
                                                                        //return new guess = vec_size * new_block_size
                    DATATYPE* sub_eigval,                               //block_size
                    const DecomposeOption option){
    int vec_size = residual.ptr_map->get_global_shape()[0];
    //int block_size = residual.map.get_global_shape()[1];
    int num_eig = option.num_eigenvalues;
    //int new_block_size = block_size + num_eig;
    
    std::array<int, 2> new_guess_shape = {vec_size, num_eig};
	auto p_map_inp = guess.ptr_map->generate_map_inp();
	p_map_inp->global_shape = {vec_size, num_eig};
	auto p_new_guess_map = p_map_inp->create_map();


    //auto new_guess_map = MAPTYPE(new_guess_shape, residual.comm.get_rank(), residual.comm.get_world_size());
    DenseTensor<2, DATATYPE, mtype, device> additional_guess(residual.copy_comm(), p_new_guess_map);
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        //std::array<int, 2> array_index;
        TensorOp::copy_vectors(additional_guess, residual, num_eig);
        DATATYPE* scale_factor = malloc<DATATYPE, device>(num_eig);

        for(int index=0; index< num_eig; index++){
//            array_index = {index, index};
            //DATATYPE coeff_i = sub_eigval[index] - tensor(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
            DATATYPE coeff_i = sub_eigval[index] - operations->get_diag_element(index);
            if(abs(coeff_i) > option.preconditioner_tolerance){
                scale_factor[index] = 1.0/coeff_i;
            }
            else{
                scale_factor[index] = 0.0;
            }
        }


        TensorOp::scale_vectors(additional_guess, scale_factor);
        free<device>(scale_factor);
        auto new_guess = TensorOp::append_vectors(guess, additional_guess);

        TensorOp::orthonormalize(new_guess, "default");
        return new_guess;
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
}

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//std::unique_ptr<DecomposeResult<DATATYPE> > davidson(DenseTensor<2, DATATYPE, mtype, device>& tensor){
std::unique_ptr<DecomposeResult<DATATYPE> > davidson(TensorOperations<mtype,device>* operations, DenseTensor<2, DATATYPE, mtype, device>* eigvec){
    DecomposeOption option;
    
    //std::unique_ptr<DATATYPE[]> real_eigvals(new DATATYPE[option.num_eigenvalues]);
    //std::unique_ptr<DATATYPE[]> imag_eigvals(new DATATYPE[option.num_eigenvalues]);
    std::vector<DATATYPE> real_eigvals(option.num_eigenvalues);
    std::vector<DATATYPE> imag_eigvals(option.num_eigenvalues);

    //auto shape = tensor.map.get_global_shape();
    auto shape = operations->get_global_shape();
    assert (shape[0] == shape[1]);
    //const int vec_size = shape[0];

    //auto current_comm = tensor.copy_comm();
    //auto rank = tensor.comm.get_rank();
    //auto world_size = tensor.comm.get_world_size();
    //auto current_comm = eigvec->copy_comm();
    //auto rank = current_comm->get_rank();
    //auto world_size = current_comm->get_world_size();

/*
    // initialization of gusss vector(s), V
    std::array<int, 2> guess_shape = {vec_size,option.num_eigenvalues};
    auto guess_map = MAPTYPE(guess_shape, rank, world_size);
    DenseTensor<2, DATATYPE, mtype, device> guess(*current_comm, guess_map);
    // guess : unit vector
    for(int i=0;i<option.num_eigenvalues;i++){
        std::array<int, 2> tmp_index = {i,i};
        guess.global_set_value(tmp_index, 1.0);
    }
    // univ vector guess : do not need orthonormalize.
*/
    //eigvec is guess.

    bool return_result = false;
    int iter = 0;
    while(iter < option.max_iterations){
		if(eigvec->ptr_comm->get_rank()==0) std::cout << "iter: " <<iter << std::endl;
		auto new_guess = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > ( *eigvec);
		//DenseTensor<2, DATATYPE, mtype, device> new_guess ( *eigvec);

        //block expansion loop
        int i_block = 0;
        
        while(true){

            const int block_size = option.num_eigenvalues*(i_block+1);
            // W_iterk = A V_k
            //auto w_iter = TensorOp::matmul(tensor, *new_guess, TRANSTYPE::N, TRANSTYPE::N);
            auto w_iter = operations->matvec(*new_guess);

            //auto subspace_matrix = TensorOp::matmul(*new_guess.get(), w_iter, TRANSTYPE::T, TRANSTYPE::N);
            auto subspace_matrix = TensorOp::matmul(*new_guess, w_iter, TRANSTYPE::T, TRANSTYPE::N);



            //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
            DATATYPE* sub_eigval = malloc<DATATYPE, device>(block_size);
            auto sub_eigvec = TensorOp::diagonalize(subspace_matrix, sub_eigval);


            //calculate ritz vector
            //Ritz vector calculation, x_ki = V_k y_ki
            auto ritz_vec = TensorOp::matmul(*new_guess, sub_eigvec, TRANSTYPE::N, TRANSTYPE::N);
            //ritz vectors are new eigenvector candidates
			
            //get residual
            auto residual = calculate_residual<DATATYPE,mtype, device>(w_iter, sub_eigval, sub_eigvec, ritz_vec);
            //check convergence
            bool is_converged = check_convergence<DATATYPE,mtype,device>(residual, option.num_eigenvalues, option.tolerance);
            if(is_converged){
                return_result = true;
                //memset<DATATYPE, device>(imag_eigvals.get(), 0.0, option.num_eigenvalues);
                //memcpy<DATATYPE, device>(real_eigvals.get(), sub_eigval, option.num_eigenvalues);
                std::fill(imag_eigvals.begin(), imag_eigvals.end(), 0.0);
                real_eigvals.assign(sub_eigval, sub_eigval+option.num_eigenvalues);

                TensorOp::copy_vectors<DATATYPE, mtype, device>(*eigvec, ritz_vec, option.num_eigenvalues);
                break;
            }
            if(i_block == option.max_block-1){
                TensorOp::copy_vectors<DATATYPE, mtype, device>(*eigvec, ritz_vec, option.num_eigenvalues);
                break;
            }
            
            //preconditioning
            new_guess = std::make_unique <DenseTensor<2, DATATYPE, mtype, device> > ( preconditioner<DATATYPE,mtype,device>(operations, residual, ritz_vec, sub_eigval, option));
            free<device>(sub_eigval);
            //delete new_guess;
            //new_guess = new DenseTensor<2, DATATYPE, mtype, device>(tmp_guess);
            
            i_block++;
        }
        //delete new_guess;
        if(return_result){    
			if(eigvec->ptr_comm->get_rank()==0)     std::cout << "CONVERGED, iter = " << iter << std::endl;            
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
    //std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const int) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const int) option.num_eigenvalues,real_eigvals,imag_eigvals));
 
    return std::move(return_val);
}
}

