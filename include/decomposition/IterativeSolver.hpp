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
std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > calculate_residue( // return residue (block_size by vec_size)
    const DenseTensor<2, DATATYPE, mtype, device>& w_iter,      //vec_size by block_size
    const typename real_type<DATATYPE>::type* sub_eigval,       //block_size
    const DenseTensor<2, DATATYPE, mtype, device>& sub_eigvec,  //block_size by block_size
    const DenseTensor<2, DATATYPE, mtype, device>& ritz_vec,    //vec_size by block_size
	const int num_eigval)
{
	using TensorOp = TensorOp<mtype,device>;
    //residue, r_ki =  W_iterk y_ki - lambda_ki x_ki
    auto new_map_inp = ritz_vec.ptr_map->generate_map_inp();
    new_map_inp->global_shape = {ritz_vec.ptr_map->get_global_shape()[0], num_eigval};
    auto new_map = new_map_inp->create_map();
    auto scaled_ritz = std::make_unique< DenseTensor<2, DATATYPE, mtype, device>  > (ritz_vec.copy_comm(), new_map); // vec_size by num_eigval
    
    TensorOp::copy_vectors(*scaled_ritz, ritz_vec, num_eigval);	
    //lambda_ki x_ki
    TensorOp::scale_vectors_(*scaled_ritz, sub_eigval);

    //W_iterk y_ki - lambda_ki x_ki
    auto tmp_residue = TensorOp::matmul(w_iter, sub_eigvec);
    auto residue = std::make_unique< DenseTensor<2, DATATYPE, mtype, device> > (tmp_residue->copy_comm(), new_map);    // vec_size by num_eigval
    TensorOp::copy_vectors(*residue, *tmp_residue, num_eigval);
    
    TensorOp::add_(*residue, *scaled_ritz, -1.0);
    return residue;
}

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
bool check_convergence_residue(const DenseTensor<2, DATATYPE, mtype, device>& residue, 
                       const int num_eigenvalues, 
					   const double tolerance){
	using TensorOp = TensorOp<mtype,device>;
    using REALTYPE = typename real_type<DATATYPE>::type;

    //convergence check
    REALTYPE* norm = malloc<REALTYPE, device>(num_eigenvalues);
    TensorOp::get_norm_of_vectors(residue, norm, num_eigenvalues);
    for(int i=0;i<num_eigenvalues;i++){
		//if(residue.ptr_comm->get_rank()==0) std::cout << i << " " << norm[i] <<std::endl; 
        if(norm[i] > tolerance){
            free<device>(norm);
            return false;
        }
    }
    free<device>(norm);
    return true;
}

template<typename DATATYPE>
bool check_convergence_eigval(typename real_type<DATATYPE>::type* old_sub_eigval, typename real_type<DATATYPE>::type* sub_eigval, const int num_eigenvalues, const double tolerance, double* max_eigdiff){
    using REALTYPE = typename real_type<DATATYPE>::type;
    bool flag = true;
    max_eigdiff[0] = 0.0;
    for(int i=0;i<num_eigenvalues;i++){
        REALTYPE diff = std::abs(sub_eigval[i] - old_sub_eigval[i]);
        //update old_sub_eigval
        old_sub_eigval[i] = sub_eigval[i];
        if(diff > tolerance){
            flag = false;
        }
        if(diff > max_eigdiff[0]){
            max_eigdiff[0] =  diff;
        }
    }
    return flag;
}


template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr<DecomposeResult<DATATYPE> > davidson(const TensorOperations<DATATYPE, mtype,device>* operations, DenseTensor<2, DATATYPE, mtype, device>* eigvec, const DecomposeOption& option){

	using TensorOp = TensorOp<mtype,device>;
    using REALTYPE = typename real_type<DATATYPE>::type;

    std::vector<REALTYPE> real_eigvals(option.num_eigenvalues);
    std::vector<REALTYPE> imag_eigvals(option.num_eigenvalues);

    const auto shape = operations->get_global_shape();
    assert (shape[0] == shape[1]);
	int block_size = shape[1];

    //Define preconditioner
    auto preconditioner  = get_preconditioner<DATATYPE,mtype,device>(operations, option);

    //0th iteration.
    //eigvec is guess.
    auto new_guess = eigvec->clone();
    auto w_iter = operations->matvec(*new_guess);
    auto subspace_matrix = TensorOp::matmul(*TensorOp::conjugate(*new_guess), *w_iter, TRANSTYPE::T, TRANSTYPE::N) ;

    //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
    REALTYPE* sub_eigval = malloc<REALTYPE, device>(eigvec->ptr_map->get_global_shape(1) ) ;

    auto sub_eigvec = TensorOp::diagonalize(*subspace_matrix, sub_eigval) ;

    //old_sub_eigval stores previous eigenvalues
    REALTYPE* old_sub_eigval = malloc<REALTYPE, device>(eigvec->ptr_map->get_global_shape(1) ) ;
    //initialize old_sub_eigval
    std::fill(old_sub_eigval, old_sub_eigval+option.num_eigenvalues, 10000.0);

    //calculate ritz vector
    //Ritz vector calculation, x_ki = V_k y_ki
    auto ritz_vec = TensorOp::matmul(*new_guess, *sub_eigvec, TRANSTYPE::N, TRANSTYPE::N) ;
    
    bool return_result = false;
    //outer loop
    //1 ~ option.max_iterations th iteration
    
    const int max_print = std::min(5, option.num_eigenvalues);
    if(eigvec->ptr_comm->get_rank()==0) std::cout << "ITER / EIGVALS" << std::endl;

    for(int i_iter = 1; i_iter < option.max_iterations ; i_iter++){
        //block expansion loop
        //i_block = number of block expanded
        int i_block = 0;
        for(int i_block = 0; i_block <= option.max_block; i_block++){
			if(eigvec->ptr_comm->get_rank()==0){
                std::cout << i_iter << "-" << i_block << " : ";
                for(int i=0;i<max_print;i++){
                    std::cout << std::fixed << std::setw(9) << std::setprecision(6) << sub_eigval[i] << "\t";
                }
            }

            //using previous w_iter, sub_eigval, sub_eigvec, ritz_vec, get residue
            auto residue = calculate_residue(*w_iter, sub_eigval, *sub_eigvec, *ritz_vec, option.num_eigenvalues);
            //std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > residue = calculate_residue<DATATYPE,mtype, device>(*w_iter, sub_eigval, *sub_eigvec, *ritz_vec, option.num_eigenvalues);
            
            //check convergence
            bool is_converged = false;
            if(option.convergence_type==CONV_TYPE::Residual){
                is_converged = check_convergence_residue<DATATYPE,mtype,device>(*residue, option.num_eigenvalues, option.tolerance);
                
                std::cout << std::endl;
            }
            else if(option.convergence_type==CONV_TYPE::Eigenvalue){
                double max_eigdiff;
                is_converged = check_convergence_eigval<DATATYPE>(old_sub_eigval, sub_eigval, option.num_eigenvalues, option.tolerance, &max_eigdiff);
                if(eigvec->ptr_comm->get_rank()==0) std::cout << "MAX_DIFF (a.u.) : " << max_eigdiff << std::endl;
            }
            if(is_converged){
                return_result = true;
                real_eigvals.assign(sub_eigval, sub_eigval+option.num_eigenvalues);
                //imag_eigvals should be filled from the diagonalization result.
                //Up to now, davidson only works for symmetric matrix, so the imaginary part should be zero.
                std::fill(imag_eigvals.begin(), imag_eigvals.end(), 0.0);

                TensorOp::copy_vectors(*eigvec, *ritz_vec, option.num_eigenvalues);
                break;
            }

            //block expansion starts
            //int block_size = option.num_eigenvalues*(i_block+1);

            if(i_block == option.max_block){
                //block_size = option.num_eigenvalues;
                TensorOp::copy_vectors(*eigvec, *ritz_vec, option.num_eigenvalues);
                new_guess = eigvec->clone();
            }
            else{
                //preconditioning, expanding the vector space which new_guess expands
                new_guess = TensorOp::append_vectors(*new_guess, *preconditioner->call(*residue, sub_eigval) );
                //orthonormalization
                TensorOp::orthonormalize(*new_guess, "qr");
            }
            // W_iterk = A V_k
            w_iter = operations->matvec(*new_guess) ;
            subspace_matrix = TensorOp::matmul(*TensorOp::conjugate(*new_guess), *w_iter, TRANSTYPE::T, TRANSTYPE::N) ;
            
            //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
            free<device>(sub_eigval);
            sub_eigval =  malloc<REALTYPE, device>(new_guess->ptr_map->get_global_shape(1) );
            sub_eigvec =  TensorOp::diagonalize(*subspace_matrix, sub_eigval) ;
            
            //calculate ritz vector
            //Ritz vector calculation, x_ki = V_k y_ki
            ritz_vec = TensorOp::matmul(*new_guess, *sub_eigvec, TRANSTYPE::N, TRANSTYPE::N) ;
        }
        if(return_result){    
            if(eigvec->ptr_comm->get_rank()==0)     std::cout << "CONVERGED, iter = " << i_iter << std::endl;            
            break;
        }
        else{
            //i_iter++;
            
        }
    }
    if(!return_result){
        if(eigvec->ptr_comm->get_rank()==0) std::cout << "NOT CONVERGED!" << std::endl;
        free<device>(sub_eigval);
        exit(-1);
    }
    free<device>(sub_eigval);
    std::unique_ptr<DecomposeResult<DATATYPE> > return_val(new DecomposeResult<DATATYPE>( (const int) option.num_eigenvalues,real_eigvals,imag_eigvals));
 
    return std::move(return_val);
}
}

