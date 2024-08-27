#pragma once
#include <memory>
#include "DecomposeOption.hpp"
#include "../Utility.hpp"
#include "../device/TensorOp.hpp"
#include "TensorOperations.hpp"
#include "DecomposeResult.hpp"
#include "Preconditioner.hpp"


namespace SE{
template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > calculate_residual( // return residual (block_size by vec_size)
    const DenseTensor<2, DATATYPE, mtype, device>& w_iter,      //vec_size by block_size
    const DATATYPE* sub_eigval,                                   //block_size
    const DenseTensor<2, DATATYPE, mtype, device>& sub_eigvec,  //block_size by block_size
    const DenseTensor<2, DATATYPE, mtype, device>& ritz_vec)    //vec_size by block_size
{
    //int block_size = sub_eigvec.map.get_global_shape()[0];
    //int vec_size = ritz_vec.map.get_global_shape()[0];
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    DenseTensor<2, DATATYPE, mtype, device> scaled_ritz(ritz_vec);
    TensorOp::scale_vectors_(scaled_ritz, sub_eigval);
    //W_iterk y_ki - lambda_ki x_ki
    auto tmp_residual = TensorOp::matmul(w_iter, sub_eigvec);
    auto residual = TensorOp::add(tmp_residual, scaled_ritz, -1.0);
    return residual;
}

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
bool check_convergence(const DenseTensor<2, DATATYPE, mtype, device>& residual, 
                       const int num_eigenvalues, 
					   const double tolerance){
    //convergence check
    double* norm = malloc<DATATYPE, device>(num_eigenvalues);
    TensorOp::get_norm_of_vectors(residual, norm, num_eigenvalues);
    for(int i=0;i<num_eigenvalues;i++){
		//if(residual.ptr_comm->get_rank()==0) std::cout << i << " " << norm[i] <<std::endl; 
        if(norm[i] > tolerance){
            free<device>(norm);
            return false;
        }
    }
    free<device>(norm);
    return true;
}


template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
//std::unique_ptr<DecomposeResult<DATATYPE> > davidson(DenseTensor<2, DATATYPE, mtype, device>& tensor){
std::unique_ptr<DecomposeResult<DATATYPE> > davidson(const TensorOperations<mtype,device>* operations, DenseTensor<2, DATATYPE, mtype, device>* eigvec, const DecomposeOption& option){
    //DecomposeOption option;
    
    std::vector<DATATYPE> real_eigvals(option.num_eigenvalues);
    std::vector<DATATYPE> imag_eigvals(option.num_eigenvalues);

    auto shape = operations->get_global_shape();
    assert (shape[0] == shape[1]);
    //eigvec is guess.

    //Define preconditioner
    auto preconditioner  = get_preconditioner<DATATYPE,mtype,device>(operations, option);

    //0th iteration.
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > new_guess = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > ( *eigvec);
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > w_iter = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (operations->matvec(*new_guess));
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > subspace_matrix = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::matmul(*new_guess, *w_iter, TRANSTYPE::T, TRANSTYPE::N) );

    //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
    DATATYPE* sub_eigval = malloc<DATATYPE, device>(option.num_eigenvalues) ;
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > sub_eigvec = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::diagonalize(*subspace_matrix, sub_eigval) );

    //calculate ritz vector
    //Ritz vector calculation, x_ki = V_k y_ki
    //ritz vectors are new eigenvector candidates
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > ritz_vec = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::matmul(*new_guess, *sub_eigvec, TRANSTYPE::N, TRANSTYPE::N) );
    
    
    bool return_result = false;
    //outer loop
    //1 ~ option.max_iterations th iteration
    for(int i_iter = 1; i_iter < option.max_iterations ; i_iter++){
        //block expansion loop
        //i_block = number of block expanded
        int i_block = 0;
        for(int i_block = 0; i_block <= option.max_block; i_block++){
			//if(eigvec->ptr_comm->get_rank()==0) std::cout << i_iter << " " << i_block <<std::endl; 
            //using previous w_iter, sub_eigval, sub_eigvec, ritz_vec, get residual
            std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > residual = calculate_residual<DATATYPE,mtype, device>(*w_iter, sub_eigval, *sub_eigvec, *ritz_vec);

            //using previous w_iter, sub_eigval, sub_eigvec, ritz_vec, get residual
            //check convergence
            bool is_converged = check_convergence<DATATYPE,mtype,device>(*residual, option.num_eigenvalues, option.tolerance);
            if(is_converged){
                return_result = true;
                real_eigvals.assign(sub_eigval, sub_eigval+option.num_eigenvalues);
                //imag_eigvals should be filled from the diagonalization result.
                //Up to now, davidson only works for symmetric matrix, so the imaginary part should be zero.
                std::fill(imag_eigvals.begin(), imag_eigvals.end(), 0.0);

                TensorOp::copy_vectors<DATATYPE, mtype, device>(*eigvec, *ritz_vec, option.num_eigenvalues);
                break;
            }

            //block expansion starts
//            int block_size = option.num_eigenvalues*(i_block+1);

            if(i_block == option.max_block){
//                block_size = option.num_eigenvalues;
                TensorOp::copy_vectors<DATATYPE, mtype, device>(*eigvec, *ritz_vec, option.num_eigenvalues);
                new_guess = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > ( *eigvec);
            }
            else{
                //preconditioning
                new_guess = TensorOp::append_vectors(*ritz_vec, *preconditioner->call(*residual, option.num_eigenvalues, sub_eigval) );
//                block_size = option.num_eigenvalues*(i_block+2);
                TensorOp::orthonormalize(*new_guess, "default");
            }
            // W_iterk = A V_k
            w_iter = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > ( operations->matvec(*new_guess) );
            subspace_matrix = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::matmul(*new_guess, *w_iter, TRANSTYPE::T, TRANSTYPE::N) );

            //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
            
            free<device>(sub_eigval);
            sub_eigval =  malloc<DATATYPE, device>(new_guess->ptr_map->get_global_shape(1) );
            sub_eigvec = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::diagonalize(*subspace_matrix, sub_eigval) );
            
            //calculate ritz vector
            //Ritz vector calculation, x_ki = V_k y_ki
            ritz_vec = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (TensorOp::matmul(*new_guess, *sub_eigvec, TRANSTYPE::N, TRANSTYPE::N) );
        }
        if(return_result){    
            if(eigvec->ptr_comm->get_rank()==0)     std::cout << "CONVERGED, iter = " << i_iter << std::endl;            
            free<device>(sub_eigval);
            break;
        }
        else{
            //i_iter++;
            
        }
    }
    if(!return_result){
        std::cout << "NOT CONVERGED!" << std::endl;
        free<device>(sub_eigval);
        exit(-1);
    }
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const int) option.num_eigenvalues,real_eigvals,imag_eigvals));
 
    return std::move(return_val);
}
}

