#pragma once
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../ContiguousMap.hpp"
#include "../Device.hpp"
#include "Utility.hpp"
#include <memory>
#include "../device/MKL/MKLComm.hpp"
#include "../device/MPI/MPIComm.hpp"


#include "../device/MPI/LinearOp.hpp"

#include "DecomposeOption.hpp"

namespace SE{

template<STORETYPE storetype, typename datatype, size_t dimension, typename computEnv, typename maptype>
void calculate_Witer(Tensor<storetype, datatype, 2, computEnv, maptype>& tensor, datatype* guess, size_t n, size_t block_size, datatype* W_iter){
    //null
}
template<typename datatype, typename computEnv, typename maptype>
void calculate_Witer(Tensor<STORETYPE::COO, datatype, 2, computEnv, maptype>& tensor, datatype* guess, size_t n, size_t block_size, datatype* W_iter){
    for(size_t i=0;i<n;i++){
        for(size_t vector_index = 0; vector_index < block_size ; vector_index++){
            W_iter[i+vector_index*n] = 0;
        }
    }
    for(auto entity : tensor.data){
        for(size_t vector_index = 0; vector_index < block_size ; vector_index++){
            W_iter[entity.first[0] + vector_index*n] += entity.second * guess[entity.first[1] + vector_index*n];
        }
    }
}
template<>
void calculate_Witer(Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >& tensor, double* guess, size_t n, size_t block_size, double* W_iter){
    size_t my_rank = tensor.comm->rank;
    size_t world_size = tensor.comm->world_size;
    /*
    size_t chunk_size = tensor.map->calculate_chunk_size(n, world_size);
    int* local_matrix_size = malloc<int, MKL>(world_size);
    //int* idisp = malloc<int, MKL>(world_size);
    //idisp[0] = 0;
    for(int rank=0;rank<world_size;rank++){
        local_matrix_size[rank] = chunk_size;
        //idisp[rank+1] = idisp[rank] + local_matrix_size[rank];
    }
    if(n % world_size != 0){
        local_matrix_size[world_size-1] = n - chunk_size * (world_size-1);
    }
    */
    size_t* local_matrix_size = tensor.map->get_partition_size_array(0);
    for(size_t i=0;i<local_matrix_size[my_rank];i++){
        for(size_t vector_index = 0; vector_index < block_size ; vector_index++){
            W_iter[i+vector_index*local_matrix_size[my_rank]] = 0;
        }
    }
    for(auto entity : tensor.data){
        for(size_t vector_index = 0; vector_index < block_size ; vector_index++){
            std::array<size_t, 2> local_index = tensor.map->get_local_array_index(entity.first, 0, tensor.comm->rank);
            W_iter[local_index[0] + vector_index*local_matrix_size[my_rank]] += entity.second * guess[tensor.map->get_global_index(local_index[1] + vector_index*local_matrix_size[my_rank], 0, my_rank)];
            
                
            std::cout << tensor.comm->rank << " entity.first : " << entity.first[0] << " " << entity.first[1] << " -> " << local_index[0] << ", " << local_index[1] << ", " << entity.second << " --> ";
            std::cout << entity.second << " * " << guess[tensor.map->get_global_index(local_index[1] + vector_index*local_matrix_size[my_rank], 0, my_rank)] <<"(";
            std::cout << tensor.map->get_global_index(local_index[1] + vector_index*local_matrix_size[my_rank], 0, my_rank) << ") = " << W_iter[local_index[0] + vector_index*local_matrix_size[my_rank]] << std::endl;
            
        }
    }
}
template<typename datatype, typename computEnv, typename maptype>
void calculate_Witer(Tensor<STORETYPE::Dense, datatype, 2, computEnv, maptype>& tensor, datatype* guess, size_t n, size_t block_size, datatype* W_iter){
    gemm<datatype, computEnv>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, n, 1.0, tensor.data, n, guess, n, 0.0, W_iter, n);
}

template<typename datatype, typename computEnv>
void subspace_diagonalization(datatype* rayleigh, size_t block_size, datatype* sub_eigval, datatype* sub_eigvec){
    datatype* rayleigh_eigval_imag = malloc<datatype, computEnv>(block_size);
    datatype* rayleigh_eigvec_left = malloc<datatype, computEnv>(block_size * block_size);

    //get eigenpair of Rayleigh matrix (lambda_ki, y_ki) of H_k
    geev<datatype, computEnv>(SE_layout::ColMajor, 'N', 'V', block_size, rayleigh, block_size, sub_eigval, rayleigh_eigval_imag, rayleigh_eigvec_left, block_size, sub_eigvec, block_size);
    free<datatype, computEnv>(rayleigh_eigval_imag);
    free<datatype, computEnv>(rayleigh_eigvec_left);
    eigenvec_sort<datatype, computEnv>(sub_eigval, sub_eigvec, block_size, block_size);
}

template<typename datatype, typename computEnv>
void calculate_ritz_vector(datatype* guess, datatype* sub_eigvec, size_t n, size_t block_size, datatype* ritz_vec){
    //Ritz vector calculation, x_ki = V_k y_ki
    gemm<datatype, computEnv>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, guess, n, sub_eigvec, block_size, 0.0, ritz_vec, n); 
}
template<>
void calculate_ritz_vector<double, MPI>(double* guess, double* sub_eigvec, size_t n, size_t block_size, double* ritz_vec){
    //Ritz vector calculation, x_ki = V_k y_ki
    gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, guess, n, sub_eigvec, block_size, 0.0, ritz_vec, n); 
}

template<typename datatype, typename computEnv>
void calculate_residual(datatype* W_iter, datatype* sub_eigval, datatype* sub_eigvec, datatype* ritz_vec, size_t n, size_t block_size, datatype* residual){
    std::cout << "not implemented" << std::endl;
}
template<>
void calculate_residual<double, MKL>(double* W_iter, double* sub_eigval, double* sub_eigvec, double* ritz_vec, size_t n, size_t block_size, double* residual){
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    for(int index = 0; index < n*block_size; index++){
        residual[index] = ritz_vec[index] * sub_eigval[index/n];
    }
    //W_iterk y_ki - lambda_ki x_ki
    gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, W_iter, n, sub_eigvec, block_size, -1.0, residual, n);
}
template<>
void calculate_residual<double, MPI>(double* W_iter, double* sub_eigval, double* sub_eigvec, double* ritz_vec, size_t n, size_t block_size, double* residual){
    //residual, r_ki =  W_iterk y_ki - lambda_ki x_ki
    //lambda_ki x_ki
    for(int index = 0; index < n*block_size; index++){
        residual[index] = ritz_vec[index] * sub_eigval[index/n];
    }
    //W_iterk y_ki - lambda_ki x_ki
    gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::NoTrans, SE_transpose::NoTrans, n, block_size, block_size, 1.0, W_iter, n, sub_eigvec, block_size, -1.0, residual, n);
}


template<typename datatype, typename computEnv>
bool check_convergence(Comm<computEnv>* _comm, datatype* residual, datatype* old_residual, size_t n, size_t num_eigenvalues, datatype tolerance){
    std::cout << "not implemented" << std::endl;
    exit(1);
}
template<>
bool check_convergence<double, MKL>(Comm<MKL>* _comm, double* residual, double* old_residual, size_t n, size_t num_eigenvalues, double tolerance){
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
template<>
bool check_convergence<double, MPI>(Comm<MPI>* _comm, double* residual, double* old_residual, size_t n, size_t num_eigenvalues, double tolerance){
    //convergence check
    double sum_of_norm_square = 0.0;
    for(int index = 0; index < n*num_eigenvalues; index++){
        sum_of_norm_square += (residual[index] - old_residual[index])*(residual[index] - old_residual[index]);
    }
    _comm->barrier();
    //MPI_Barrier(MPI_COMM_WORLD);
    double total_sum_of_norm_square;
    _comm->allreduce(&sum_of_norm_square, 1, &total_sum_of_norm_square, SEop::SUM);
    std::cout << "sum_of_norm_square : " << total_sum_of_norm_square << std::endl;
    return total_sum_of_norm_square < tolerance*tolerance;
}

template<STORETYPE storetype, typename datatype, size_t dimension, typename computEnv, typename maptype>
void preconditioner(Tensor<storetype, datatype, dimension, computEnv, maptype>& tensor, DecomposeOption option, datatype* sub_eigval, datatype* residual, size_t block_size, datatype* guess){
    std::cout << "undefined Tensor?" << std::endl;
}

template <typename datatype, typename maptype>
void preconditioner(Tensor<STORETYPE::Dense, datatype, 2, MKL, maptype>& tensor, DecomposeOption option, datatype* sub_eigval, datatype* residual, size_t block_size, datatype* guess){
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = tensor.shape[0];
        std::array<size_t, 2> index;
        //std::cout << "precond: ";
        for(size_t i=0; i<option.num_eigenvalues; i++){
            index = {i,i};
            datatype coeff_i = sub_eigval[i] - tensor.data[i+n*i];
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
template <typename datatype, typename maptype>
void preconditioner(Tensor<STORETYPE::COO, datatype, 2, MKL, maptype>& tensor, DecomposeOption option, datatype* sub_eigval, datatype* residual, size_t block_size, datatype* guess){
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = tensor.shape[0];
        std::array<size_t, 2> index;
        //std::cout << "precond: ";
        for(size_t i=0; i<option.num_eigenvalues; i++){
            index = {i,i};
            datatype coeff_i = sub_eigval[i] - tensor.operator()(index);
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
template <>
void preconditioner(Tensor<STORETYPE::Dense, double, 2, MPI, ContiguousMap<2> >& tensor, DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess){
    // sparse tensor는 나눠서 저장
    // sub_eigval은 모두가 들고 있음
    // residual은 나눠서 저장
    // block_size는 모두가 알고있음
    // guess는 나눠서 저장

    size_t my_rank = tensor.comm->rank;
    size_t world_size = tensor.comm->world_size;
    
    if(option.preconditioner == PRECOND_TYPE::Diagonal){
        size_t n = tensor.shape[0];
        
        size_t chunk_size = tensor.shape[0] / world_size;// tensor.map->calculate_chunk_size(n, world_size);
        /*
        int* local_matrix_size = malloc<int, MKL>(world_size);
        for(int rank=0;rank<world_size;rank++){
            local_matrix_size[rank] = chunk_size;
            //idisp[rank+1] = idisp[rank] + local_matrix_size[rank];
        }
        if(n % world_size != 0){
            local_matrix_size[world_size-1] = n - chunk_size * (world_size-1);
        }
        */
        size_t* local_matrix_size = tensor.map->get_partition_size_array(0);

        size_t max_PID = (option.num_eigenvalues -1) / chunk_size;
        std::array<size_t, 2> index;
        double* local_coeff_i = malloc<double, MPI>(chunk_size);

        size_t* number_of_coeff_i = malloc<size_t, MKL>(world_size);
        //size_t* idisp = malloc<size_t, MKL>(world_size);
        memset<size_t, MKL>(number_of_coeff_i, 0, world_size);
        //memset<size_t, MKL>(idisp, 0, world_size);


        int tmp_num_eig = option.num_eigenvalues;
        //idisp[0] = 0;
        for(int rank=0;rank<=max_PID;rank++){
            if(tmp_num_eig > chunk_size){
                number_of_coeff_i[rank] = chunk_size;
                tmp_num_eig -= chunk_size;
            }
            else{
                number_of_coeff_i[rank] = tmp_num_eig;
            }
            //idisp[rank+1] = idisp[rank] + number_of_coeff_i[rank];
        }

        if(my_rank <= max_PID){
            for(size_t i=0; i<chunk_size; i++){
                if(i + my_rank*chunk_size == n) break;
                index = {i + my_rank*chunk_size, i + my_rank*chunk_size};
                local_coeff_i[i] = sub_eigval[i + my_rank*chunk_size] - tensor.operator()(index);
                //number_of_coeff_i[my_rank]++;
            }
        }
        double* coeff_i = malloc<double, MKL>(option.num_eigenvalues);
        
        tensor.comm->allgatherv(local_coeff_i, number_of_coeff_i[my_rank], coeff_i, number_of_coeff_i);
        
        //MPI_Allgatherv(local_coeff_i, number_of_coeff_i[my_rank], MPI_DOUBLE, coeff_i, number_of_coeff_i, idisp, MPI_DOUBLE, tensor.comm->mpi_comm);
        //std::cout << "precond: ";
        for(size_t i=0; i<option.num_eigenvalues; i++){
            
            if(coeff_i[i] > option.preconditioner_tolerance){
                for(int j=0;j<local_matrix_size[my_rank];j++){
                    guess[local_matrix_size[my_rank]*(block_size+i) + j] = residual[local_matrix_size[my_rank]*i + j] / coeff_i[i];
                    //std::cout << "( " << local_matrix_size[my_rank]*(block_size+i) + j << " : " << residual[local_matrix_size[my_rank]*i+j] / coeff_i[i] << "), ";
                }
            }
        }
    }
    else{
        std::cout << "not implemented" << std::endl;
        exit(1);
    }
}

template <typename datatype, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype> > davidson(Tensor<STORETYPE::Dense, datatype, 2, computEnv, maptype>& tensor){
    DecomposeOption option;
    std::unique_ptr<datatype[]> real_eigvals(new datatype[option.num_eigenvalues]);
    std::unique_ptr<datatype[]> imag_eigvals(new datatype[option.num_eigenvalues]);

    assert(tensor.shape[0] == tensor.shape[1]);
    const size_t n = tensor.shape[0];

    //std::unique_ptr<datatype[]> eigvec_0(new datatype[option.num_eigenvalues*n]);

    int block_size = option.num_eigenvalues;
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }
    // initialization of gusss vector(s), V
    // guess : unit vector
    datatype* guess = malloc<datatype, computEnv>(n*block_size*option.max_iterations);
    memset<datatype, computEnv>(guess, 0.0, n*block_size*option.max_iterations);
    for(int i=0;i<option.num_eigenvalues;i++){
        guess[i*n+i] = 1.0;
    }
    datatype* old_residual = malloc<datatype, computEnv>(n*option.num_eigenvalues);
    memset<datatype, computEnv>(old_residual, 0.0, n*option.num_eigenvalues);
    
    int iter = 0;
    while(iter<option.max_iterations){
        orthonormalize<datatype, computEnv>(guess, n, block_size, "qr");
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        datatype* W_iter = malloc<datatype, computEnv>(n*block_size);
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
        datatype* rayleigh = malloc<datatype, computEnv>(block_size* block_size);
        datatype* rayleigh_eigval_0 = malloc<datatype, computEnv>(block_size);
        datatype* rayleigh_eigvec_0 = malloc<datatype, computEnv>(block_size * block_size);
        
        //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
        gemm<datatype, computEnv>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
/*
    std::cout << "rayleigh" << std::endl;
    for(int i=0;i<block_size;i++){
        for(int j=0;j<block_size;j++){
            std::cout << std::setw(6) << rayleigh[i+j*block_size] << " ";
        }
        std::cout << std::endl;
    }
*/
        subspace_diagonalization<datatype, computEnv>(rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigvec_0);

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
        datatype* ritz_vec = malloc<datatype, computEnv>(n*block_size);
        calculate_ritz_vector<datatype, computEnv>(guess, rayleigh_eigvec_0, n, block_size, ritz_vec);
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
        datatype* residual = malloc<datatype, computEnv>(n*block_size);
        calculate_residual<datatype, computEnv>(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<datatype, computEnv>(W_iter);

        bool is_converged = check_convergence<datatype, computEnv>(tensor.comm, residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            //memcpy<datatype, computEnv>(eigvec_0.get(), ritz_vec, n*option.num_eigenvalues);
            memcpy<datatype, computEnv>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
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
        memcpy<datatype, computEnv>(old_residual, residual, n*option.num_eigenvalues);
        block_size += option.num_eigenvalues;
        iter++;
        free<datatype, computEnv>(rayleigh_eigval_0);
        free<datatype, computEnv>(rayleigh_eigvec_0);
        free<datatype, computEnv>(ritz_vec);
        free<datatype, computEnv>(residual);
    }
    if(iter == option.max_iterations){
        std::cout << "diagonalization did not converged!" << std::endl;
        exit(-1);
    }
    else{
        for(int i=0;i<option.num_eigenvalues;i++){
            imag_eigvals.get()[i] = 0;
        }
    
        std::unique_ptr<DecomposeResult<datatype> > return_val(new DecomposeResult<datatype>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));

        return std::move(return_val);
    }
}
template <typename datatype, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype> > davidson(Tensor<STORETYPE::COO, datatype, 2, computEnv, maptype>& tensor){
    DecomposeOption option;
    std::unique_ptr<datatype[]> real_eigvals(new datatype[option.num_eigenvalues]);
    std::unique_ptr<datatype[]> imag_eigvals(new datatype[option.num_eigenvalues]);

    assert(tensor.shape[0] == tensor.shape[1]);
    const size_t n = tensor.shape[0];

    //std::unique_ptr<datatype[]> eigvec_0(new datatype[option.num_eigenvalues*n]);

    int block_size = option.num_eigenvalues;
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }
    // initialization of gusss vector(s), V
    // guess : unit vector
    datatype* guess = malloc<datatype, computEnv>(n*block_size*option.max_iterations);
    memset<datatype, computEnv>(guess, 0.0, n*block_size);
    for(int i=0;i<option.num_eigenvalues;i++){
        guess[i*n+i] = 1.0;
    }
    datatype* old_residual = malloc<datatype, computEnv>(n*option.num_eigenvalues);
    memset<datatype, computEnv>(old_residual, 0.0, n*option.num_eigenvalues);

    int iter = 0;
    while(iter < option.max_iterations){
        orthonormalize<datatype, computEnv>(guess, n, block_size, "qr");
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        datatype* W_iter = malloc<datatype, computEnv>(n*block_size);
        calculate_Witer(tensor, guess, n, block_size, W_iter);
        
        datatype* rayleigh = malloc<datatype, computEnv>(block_size* block_size);
        datatype* rayleigh_eigval_0 = malloc<datatype, computEnv>(block_size);
        datatype* rayleigh_eigvec_0 = malloc<datatype, computEnv>(block_size * block_size);

        //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
        gemm<datatype, computEnv>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, block_size, block_size, n, 1.0, guess, n, W_iter, n, 0.0, rayleigh, block_size);
        subspace_diagonalization<datatype, computEnv>(rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigvec_0);

        //calculate ritz vector
        datatype* ritz_vec = malloc<datatype, computEnv>(n*block_size);
        calculate_ritz_vector<datatype, computEnv>(guess, rayleigh_eigvec_0, n, block_size, ritz_vec);

        datatype* residual = malloc<datatype, computEnv>(n*block_size);
        calculate_residual<datatype, computEnv>(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, n, block_size, residual);
        free<datatype, computEnv>(W_iter);

        bool is_converged = check_convergence<datatype, computEnv>(tensor.comm, residual, old_residual, n, option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            //memcpy<datatype, computEnv>(eigvec_0.get(), ritz_vec, n*option.num_eigenvalues);
            memcpy<datatype, computEnv>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }
        preconditioner(tensor, option,rayleigh_eigval_0, residual, block_size, guess);
        
//        std::cout << std::endl;
        memcpy<datatype, computEnv>(old_residual, residual, n*option.num_eigenvalues);
        block_size += option.num_eigenvalues;
        iter++;

        free<datatype, computEnv>(rayleigh_eigval_0);
        free<datatype, computEnv>(rayleigh_eigvec_0);
        free<datatype, computEnv>(ritz_vec);
        free<datatype, computEnv>(residual);
    }
    if(iter == option.max_iterations){
        std::cout << "diagonalization did not converged!" << std::endl;
        exit(-1);
    }
    else{
        for(int i=0;i<option.num_eigenvalues;i++){
            imag_eigvals.get()[i] = 0;
        }
    
        std::unique_ptr<DecomposeResult<datatype> > return_val(new DecomposeResult<datatype>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));

        return std::move(return_val);
    }
}

template <>
std::unique_ptr<DecomposeResult<double> > davidson(Tensor<STORETYPE::COO, double, 2, MPI, ContiguousMap<2> >& tensor){
    DecomposeOption option;
    std::unique_ptr<double[]> real_eigvals(new double[option.num_eigenvalues]);
    std::unique_ptr<double[]> imag_eigvals(new double[option.num_eigenvalues]);

    assert(tensor.shape[0] == tensor.shape[1]);
    const size_t n = tensor.shape[0];

    int block_size = option.num_eigenvalues;
    if(option.max_iterations * block_size > n){
        std::cout << "max iteration number " << option.max_iterations << " is too large!" << std::endl;
        option.max_iterations = n/block_size;
        std::cout << "max_iterateion is changed to " << option.max_iterations << std::endl;
    }

    // initialization of gusss vector(s), V
    // guess : unit vector
    // row-wise partitioning
    size_t my_rank = tensor.comm->rank;
    size_t world_size = tensor.comm->world_size;
    /*
    size_t chunk_size = tensor.map->calculate_chunk_size(n, world_size);
    int* local_matrix_size = malloc<int, MKL>(world_size);
    //int* idisp = malloc<int, MKL>(world_size);
    //idisp[0] = 0;
    for(int rank=0;rank<world_size;rank++){
        local_matrix_size[rank] = chunk_size;
        //idisp[rank+1] = idisp[rank] + local_matrix_size[rank];
    }
    if(n % world_size != 0){
        local_matrix_size[world_size-1] = n - chunk_size * (world_size-1);
    }
    */
    size_t chunk_size = tensor.shape[0] / world_size;// tensor.map->calculate_chunk_size(n, world_size);
    /*
    int* local_matrix_size = malloc<int, MKL>(world_size);
    for(int rank=0;rank<world_size;rank++){
        local_matrix_size[rank] = chunk_size;
        //idisp[rank+1] = idisp[rank] + local_matrix_size[rank];
    }
    if(n % world_size != 0){
        local_matrix_size[world_size-1] = n - chunk_size * (world_size-1);
    }
    */
    size_t* local_matrix_size = tensor.map->get_partition_size_array(0);

    double* guess = malloc<double, MPI>(local_matrix_size[my_rank]*block_size*option.max_iterations);
    memset<double, MPI>(guess, 0.0, local_matrix_size[my_rank]*block_size*option.max_iterations);
    for(int i=0;i<option.num_eigenvalues;i++){
        //std::cout << "i = " << i << " , rank = " << my_rank << " , bla = " << tensor.map->get_my_rank_from_global_index(i*n+i, 0, world_size) << std::endl;
        if(tensor.map->get_my_rank_from_global_index(i, 0) == my_rank){
            guess[tensor.map->get_local_index(i*n+i,0,my_rank)] = 1.0;
            //std::cout << "i*n+i, i = " << i << ", " << my_rank << std::endl;
        }
    }
    double* old_residual = malloc<double, MPI>(local_matrix_size[my_rank]*option.num_eigenvalues);
    memset<double, MPI>(old_residual, 0.0, local_matrix_size[my_rank]*option.num_eigenvalues);

    //eigvec initialization
    //no partitioning
    //std::unique_ptr<double[]> eigvec_0(new double[option.num_eigenvalues*n]);

    //for(int j=0;j<option.num_eigenvalues;j++){
    //for(int i=0;i<local_matrix_size[my_rank];i++){
    std::cout << "guess : rank : " << my_rank << " : ";
    for(int i=0;i<local_matrix_size[my_rank]*block_size;i++){
        //std::cout << guess[i+local_matrix_size[my_rank]*j] ;
        std::cout << guess[i] << ' ';
    }

    std::cout << std::endl;//'\t' << guess_gather[tensor.map->get_global_index(i+local_matrix_size[my_rank]*j,0,my_rank,world_size)] << std::endl;


    int iter = 0;
    while(iter < option.max_iterations){
        //parallel orthonormalization is not implemented.
        double* guess_gather = malloc<double, MKL>(n*block_size); // 모든 process에서 동일하게 들고있음
        //memset<double, MKL>(guess_gather, 0.0, n*block_size);
        for(int guess_num = 0; guess_num<block_size; guess_num++){
            //std::cout << "guessnum = " << guess_num << std::endl;
            //std::cout << "rank = " << my_rank << " , a: " << guess_num*local_matrix_size[my_rank] << "->" << guess[guess_num*local_matrix_size[my_rank]] << ", " << local_matrix_size[my_rank] << ", " << guess_num*n << std::endl;
            tensor.comm->barrier();
            tensor.comm->allgatherv(&guess[guess_num*local_matrix_size[my_rank]], local_matrix_size[my_rank], &guess_gather[guess_num*n], local_matrix_size);
            //MPI_Allgatherv(guess[guess_num*local_matrix_size[my_rank]], local_matrix_size[my_rank], MPI_DOUBLE, guess_gather[guess_num*n], local_matrix_size, idisp, MPI_DOUBLE, comm->mpi_comm);
        }
        orthonormalize<double, MKL>(guess_gather, n, block_size, "qr");
        std::cout << "guess_gather[0,0] = " << guess_gather[0] << std::endl;
        std::cout << "end of ortho" << std::endl;
        
        // guess_gather를 guess로 재분배.
        for(int guess_num = 0; guess_num<block_size; guess_num++){
            tensor.comm->scatterv(&guess_gather[guess_num*n], local_matrix_size, &guess[guess_num*local_matrix_size[my_rank]], local_matrix_size[my_rank], 0);
        }
        tensor.comm->barrier();
    std::cout << "guess : rank : " << my_rank << " : ";
    for(int i=0;i<local_matrix_size[my_rank]*block_size;i++){
        //std::cout << guess[i+local_matrix_size[my_rank]*j] ;
        std::cout << guess[i] << ' ';
    }
        std::cout << "end of scatterv" << std::endl;
        
        // W_iterk = A V_k
        // W_iter should be updated everytime because of the numerical instability
        double* W_iter = malloc<double, MPI>(local_matrix_size[my_rank]*block_size);
        //calculate_Witer(tensor, guess_gather, local_matrix_size[my_rank], block_size, W_iter);
         std::cout << "guessgather : rank : " << my_rank << " : ";
    for(int i=0;i<n*block_size;i++){
        //std::cout << guess[i+local_matrix_size[my_rank]*j] ;
        std::cout << guess_gather[i] << ' ';
    }
    std::cout << std::endl;
        calculate_Witer(tensor, guess_gather, n, block_size, W_iter);
        std::cout << "end of witer" << std::endl;
        //Subspace(Rayleigh) matrix (dense) H_k = V_k^t A V_k
        double* rayleigh = malloc<double, MKL>(block_size* block_size);

        double* temp_c = malloc<double, MKL>(block_size*block_size);
        tensor.comm->barrier();
        std::cout << my_rank << " guess: ";
        for(int i=0;i<local_matrix_size[my_rank]*block_size;i++){
            std::cout << guess[i] << " ";
        }
        std::cout << std::endl;
        std::cout << my_rank << " Witer: ";
        for(int i=0;i<local_matrix_size[my_rank]*block_size;i++){
            std::cout << W_iter[i] << " ";
        }
        std::cout << std::endl;
      

        gemm<double, MKL>(SE_layout::ColMajor, SE_transpose::Trans, SE_transpose::NoTrans, block_size, block_size, local_matrix_size[my_rank], 1.0, guess, n, W_iter, n, 0.0, temp_c, block_size);
        std::cout << my_rank << "end of gemm" << std::endl;
        tensor.comm->barrier();
        tensor.comm->allreduce(temp_c, block_size*block_size, rayleigh, SEop::SUM);
        std::cout << "end of allreduce" << std::endl;

        std::cout << my_rank << " rayleigh: ";
        for(int i=0;i<block_size*block_size;i++){
            std::cout << rayleigh[i] << " ";
        }


        //diagonalization
        double* rayleigh_eigval_0 = malloc<double, MKL>(block_size);
        double* rayleigh_eigvec_0 = malloc<double, MKL>(block_size * block_size);
        subspace_diagonalization<double, MKL>(rayleigh, block_size, rayleigh_eigval_0, rayleigh_eigvec_0);
        std::cout << "end of diag" << std::endl;
        
        std::cout << my_rank << " rayleigh: ";
        for(int i=0;i<block_size*block_size;i++){
            std::cout << rayleigh[i] << " ";
        }
        std::cout << std::endl;
        std::cout << my_rank << " diag: ";
        for(int i=0;i<block_size*block_size;i++){
            std::cout << rayleigh_eigvec_0[i] << " ";
        }
        std::cout << std::endl;
        
        //calculate ritz vector
        double* ritz_vec = malloc<double, MPI>(local_matrix_size[my_rank]*block_size);
        calculate_ritz_vector<double, MPI>(guess, rayleigh_eigvec_0, local_matrix_size[my_rank], block_size, ritz_vec);
        tensor.comm->barrier();
        std::cout << "end of ritz" << std::endl;
        
        std::cout << my_rank << " ritz: ";
        for(int i=0;i<local_matrix_size[my_rank]*block_size;i++){
            std::cout << ritz_vec[i] << " ";
        }
        std::cout << std::endl;
        
        //calculate residual
        double* residual = malloc<double, MPI>(local_matrix_size[my_rank]*block_size);
        calculate_residual<double, MPI>(W_iter, rayleigh_eigval_0, rayleigh_eigvec_0, ritz_vec, local_matrix_size[my_rank], block_size, residual);
        tensor.comm->barrier();
        free<double, MPI>(W_iter);
        //std::cout << "end of residual" << std::endl;
        bool is_converged = check_convergence<double, MPI>(tensor.comm, residual, old_residual, local_matrix_size[my_rank], option.num_eigenvalues, option.tolerance);
        if(iter != 0 && is_converged){
            //memcpy<double, MPI>(eigvec_0.get(), ritz_vec, n*option.num_eigenvalues);
            memcpy<double, MKL>(real_eigvals.get(), rayleigh_eigval_0, option.num_eigenvalues);
            break;
        }
        //correction vector
        //Using diagonal preconditioner
        if(block_size > n-option.num_eigenvalues){
            printf( "The algorithm failed to compute eigenvalues.\n" );
            exit( 1 );
        }
        preconditioner(tensor, option,rayleigh_eigval_0, residual, block_size, guess);

         std::cout << "\n" << my_rank << " guess: ";
        for(int i=0;i<local_matrix_size[my_rank]*(block_size+1);i++){
            std::cout << std::setw(3) << guess[i] << " ";
        }
        std::cout << std::endl;

        memcpy<double, MPI>(old_residual, residual, local_matrix_size[my_rank]*option.num_eigenvalues);
        block_size += option.num_eigenvalues;
        iter++;

        free<double, MKL>(rayleigh_eigval_0);
        free<double, MKL>(rayleigh_eigvec_0);
        free<double, MPI>(ritz_vec);
        free<double, MPI>(residual);
        tensor.comm->barrier();
    }
    if(iter == option.max_iterations){
        std::cout << "diagonalization did not converged!" << std::endl;
        exit(-1);
    }
    else{
        for(int i=0;i<option.num_eigenvalues;i++){
            imag_eigvals.get()[i] = 0;
        }
    
        std::unique_ptr<DecomposeResult<double> > return_val(new DecomposeResult<double>( (const size_t) option.num_eigenvalues,std::move(real_eigvals),std::move(imag_eigvals)));

        return std::move(return_val);
    }
}



}

