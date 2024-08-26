#pragma once
#include <memory>
#include "DecomposeOption.hpp"
#include "../device/TensorOp.hpp"
#include "TensorOperations.hpp"

namespace SE{

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class Preconditioner{
public:
	Preconditioner<DATATYPE,mtype,device>(const TensorOperations<mtype, device>* operations,const DecomposeOption option, std::string type): option(option), operations(operations), type(type){
		return;
	}
	virtual ~Preconditioner() = default;
	virtual std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > call (
                                                               DenseTensor<2, DATATYPE, mtype, device>& residual, //vec_size * block_size
                                                               DATATYPE* sub_eigval) const=0;                           //block_size
															   
    friend std::ostream& operator<< (std::ostream& stream, const Preconditioner<DATATYPE, mtype, device>& preconditioner){
    //void print_tensor_info() const{
        //if(preconditioner.ptr_comm->get_rank() == 0){
		//preconditioner does not have ptr_comm
            stream << "========= Preconditioner Info =========" <<std::endl;
            stream << "type: " <<  preconditioner.type << std::endl; 
		//}
		return stream;
	}
protected:
	const DecomposeOption option;
	const TensorOperations<mtype, device>* operations;
	const std::string  type = "Base";
};

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class DiagonalPreconditioner: public Preconditioner<DATATYPE, mtype, device>{
public:
	DiagonalPreconditioner(const TensorOperations<mtype, device>* operations,const DecomposeOption option):Preconditioner<DATATYPE,mtype,device>(operations, option, "DiagonalPreconditioner"){
	}
	std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > call (DenseTensor<2, DATATYPE, mtype, device>& residual,
													   DATATYPE* sub_eigval) const{
		
	
        int vec_size = residual.ptr_map->get_global_shape()[0];
        //int block_size = residual.map.get_global_shape()[1];
        int num_eig = this->option.num_eigenvalues;
        //int new_block_size = block_size + num_eig;
        
        std::array<int, 2> new_guess_shape = {vec_size, num_eig};
    	auto p_map_inp = residual.ptr_map->generate_map_inp();
    	p_map_inp->global_shape = {vec_size, num_eig};
    	auto p_new_guess_map = p_map_inp->create_map();
    
    
        DenseTensor<2, DATATYPE, mtype, device> additional_guess(residual.copy_comm(), p_new_guess_map);
        TensorOp::copy_vectors(additional_guess, residual, num_eig);
        DATATYPE* scale_factor = malloc<DATATYPE, device>(num_eig);
    
        for(int index=0; index< num_eig; index++){
    //        array_index = {index, index};
            //DATATYPE coeff_i = sub_eigval[index] - tensor(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
            DATATYPE coeff_i = sub_eigval[index] - this->operations->get_diag_element(index);
            if(abs(coeff_i) > this->option.preconditioner_tolerance){
                scale_factor[index] = 1.0/coeff_i;
            }
            else{
                scale_factor[index] = 0.0;
            }
        }
    
    
        TensorOp::scale_vectors(additional_guess, scale_factor);
        free<device>(scale_factor);
//        auto new_guess = TensorOp::append_vectors(guess, additional_guess);
    
//        TensorOp::orthonormalize(new_guess, "default");
        return std::make_unique<DenseTensor<2, DATATYPE, mtype, device> > (additional_guess);
    }
};

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class ISI2Preconditioner: public Preconditioner<DATATYPE, mtype, device>{
public:

	ISI2Preconditioner(const TensorOperations<mtype, device>* operations,const DecomposeOption option):Preconditioner<DATATYPE,mtype,device>(operations, option, "ISI2"){
		this->pcg_precond=std::make_unique<DiagonalPreconditioner<DATATYPE,mtype,device> > (operations,option);
	}
	std::unique_ptr< DenseTensor<2, DATATYPE, mtype, device> > call (DenseTensor<2, DATATYPE, mtype, device>& residual,
													   DATATYPE* sub_eigval) const {
		


    	const int i_zero = 0;
    	const int i_one  = 1;
    	const DATATYPE one  = 1.;
    	const DATATYPE zero = 0.;
    

    	const int vec_size = residual.ptr_map->get_global_shape()[0];
    	const int num_vec = residual.ptr_map->get_global_shape()[1];

		DATATYPE* conv         = malloc<DATATYPE,device> ( num_vec );
        DATATYPE* norm2        = malloc<DATATYPE, device>( num_vec );
		DATATYPE* rzold        = malloc<DATATYPE,device> ( num_vec );
		DATATYPE* alpha        = malloc<DATATYPE,device> ( num_vec );
        DATATYPE* shift_values = malloc<DATATYPE, device>( num_vec );
		DATATYPE* rznew        = malloc<DATATYPE,device> ( num_vec );
		DATATYPE* beta         = malloc<DATATYPE,device> ( num_vec );

		// caluclate square of residual norm 
    	TensorOp::get_norm_of_vectors(residual, norm2, num_vec, false);
    
    	memcpy<DATATYPE, device>(shift_values, sub_eigval, num_vec);
    	axpy<DATATYPE, device>(num_vec, -0.1, norm2, i_one, shift_values, i_one); 
    
		bool check_initial_norm = true;
		//#pragma omp parallel for 
    	for (int i=0; i<num_vec; i++){
    		if( norm2[i] >10){
    			shift_values[i]=0;
				check_initial_norm =false;	
    		}
    	}

		auto p_i = residual.clone();
		if( check_initial_norm ==false){
			return this->pcg_precond->call( *p_i, sub_eigval) ;
		}

		////// line 7 start
    	auto r = TensorOp::scale_vectors(*p_i, shift_values);  //\tilde{epsilon} *r 
    	r = TensorOp::add<DATATYPE,mtype,device>(*r,       this->operations->matvec(*p_i),  -1.0) ; // \tilde{epsilon} *p_i -H@p_i
    	r = TensorOp::add<DATATYPE,mtype,device>(residual, *r,                           1.0) ; //r_i + \tilde{epsilon} *p_i - H@p_i
		////// line 7 end 
		
		////// line 8 start
		auto p = this->pcg_precond->call( *r, sub_eigval) ;
		////// line 8 end

		////// line 9 start
		auto z = p->clone();
		////// line 9 end

		////// line 10 start
		TensorOp::vectorwise_dot( *TensorOp::conjugate(*r), *z, rzold, num_vec );
		////// line 10 end
		
		for (int i_iter = 0; i_iter<this->option.preconditioner_max_iterations; i_iter++){
			///// line 12 start
	    	auto scaled_p = TensorOp::scale_vectors(*p, shift_values);  //\tilde{epsilon} * p
	    	auto hp = TensorOp::add<DATATYPE,mtype,device>( this->operations->matvec(*p), *scaled_p, -1.0) ; // Hp -\tilde{epsilon} *p
			///// line 12 end
			
			///// line 13 start
			TensorOp::vectorwise_dot( *TensorOp::conjugate(*p), *hp, alpha, num_vec ); 
			for (int i=0; i<num_vec; i++){
				alpha[i] = rzold[i] /alpha[i];
			}
			///// line 13 end

			///// line 14 start
			scaled_p = TensorOp::scale_vectors(*p, alpha);
			p_i = TensorOp::add(*p_i, *scaled_p, 1.0) ;
			///// line 14 end 

			///// line 15 start
			auto scaled_hp = TensorOp::scale_vectors(*hp, alpha);
			r = TensorOp::add(*r, *scaled_hp, -1.0) ;
			scaled_hp.reset();
			///// line 15 end
			
			///// line 16 end
			z = this->pcg_precond->call( *r, sub_eigval) ;
			///// line 16 end

			
			///// line 17 end
			TensorOp::vectorwise_dot( *TensorOp::conjugate(*r), *z, rznew, num_vec );
			///// line 17 end


			TensorOp::get_norm_of_vectors(*r, conv, num_vec );
			bool check_conv = true;
			for (int j =0; j<num_vec; j++){
				check_conv =   ( check_conv &&   (conv[j]<0.25*sqrt(norm2[j])  ) );
				//if(residual.ptr_comm->get_rank()==0) std::cout << std::scientific<< rzold[j] <<" " << rznew[j] << " " <<rznew[j]/rzold[j]<<std::endl;
				beta[j] = rznew[j] / rzold[j]; // beta = rznew/rzold; 
			}

			if (check_conv){
				auto residual_clone = residual.clone();
				scaled_p = TensorOp::scale_vectors(*p_i, shift_values);  //\tilde{epsilon} * p
				auto val = TensorOp::add<DATATYPE,mtype,device>( this->operations->matvec(*p_i), *scaled_p, -1.0); 
				val = TensorOp::add<DATATYPE,mtype,device>( *residual_clone, *val, -1.0 );
				TensorOp::get_norm_of_vectors(*val, norm2, num_vec);
				break;	
			}
			// line 22 start	
			TensorOp::scale_vectors_(*p, beta); //only for this line rzold is same as beta
			//TensorOp::add_(*p,*z,1.0) ; 
			//TensorOp::add_(*z,*p,1.0) ; 
			p = TensorOp::add(*p,*z,1.0);
			// line 22 end 
			//
			memcpy<DATATYPE, device> (rzold, rznew, num_vec);
		}
		free<device>(beta);
		free<device>(rznew);
		free<device>(shift_values);
		free<device>(alpha);
		free<device>(rzold);
		free<device>(norm2);
		free<device>(conv);

		TensorOp::scale_vectors_(*p_i,-1.0);
		return p_i;
    }
	protected:
		std::unique_ptr<DiagonalPreconditioner<DATATYPE, mtype, device> > pcg_precond;
};

template <typename DATATYPE, MTYPE mtype, DEVICETYPE device>					
const std::unique_ptr<Preconditioner<DATATYPE,mtype,device> > get_preconditioner(const TensorOperations<mtype, device>* operations,const DecomposeOption option){
	if(option.preconditioner== PRECOND_TYPE::Diagonal)
		return 	std::make_unique<DiagonalPreconditioner<DATATYPE,mtype,device> > ( DiagonalPreconditioner<DATATYPE,mtype,device>(operations, option) );
	else if (option.preconditioner==PRECOND_TYPE::ISI2){
		return std::make_unique<ISI2Preconditioner<DATATYPE,mtype,device> > (ISI2Preconditioner<DATATYPE,mtype,device>(operations, option) );
	}
	assert(true);
	// Default
	std::cout << "unknown preconditioner!" << std::endl;
	exit(-1);
};					

}

