#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>

#include "device/mkl/LinearOp.hpp"
#include "device/mkl/TensorOp.hpp"
#include "device/mkl/MKLComm.hpp"

#include "Contiguous1DMap.hpp"
#include "DenseTensor.hpp"
#include "SparseTensor.hpp"

#include "decomposition/Decompose.hpp"

#include "decomposition/TensorOperations.hpp"
#include "../example/TestOperations.hpp"
#include "decomposition/PyOperations.hpp"

using namespace SE;

int main(int argc, char* argv[]){
    //Initialize MKL Comm
    std::cout << "# Create Comm" << std::endl;
	MKLCommInp comm_inp;
    auto ptr_comm = comm_inp.create_comm();
    
    std::cout << "# Create Matrix" << std::endl;
    int N = 300;
    const int num_eig = 3;
    std::array<int, 2> shape = {N,N};
	Contiguous1DMapInp<2> map2_inp( shape );

    double invh2 = 1.0;
    //1D Hamiltonian-like matrix
    std::unique_ptr<double[], std::function<void(double*)> > test_data2 ( malloc<double, DEVICETYPE::MKL>(N*N), free<DEVICETYPE::MKL> );
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            test_data2.get()[i+j*N] = 0;
            if(i == j)                  test_data2.get()[i+j*N] += 2.0*((double)i-(double)N)   - invh2*5.0/2.0;
            if(i == j +1 || i == j -1)  test_data2.get()[i+j*N] += invh2*4.0/3.0;
            if(i == j +2 || i == j -2)  test_data2[i+j*N] -= invh2*1.0/12.0;
            if(i == j +3 || i == j -3)  test_data2[i+j*N] += 0.3;
            if(i == j +4 || i == j -4)  test_data2[i+j*N] -= 0.1;
            if(i !=j && i%100 == 0 && j%100 == 0)  test_data2[i+j*N] += 0.01;
        }
    }
    
    DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix2(ptr_comm, map2_inp.create_map(), std::move(test_data2));
    
    std::cout << "# Copy matrix2 into matrix3" << std::endl;
    DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix3(test_matrix2);
    
    std::cout << "========================\nDense matrix davidson test" << std::endl;
    std::array<int, 2> guess_shape = {N,num_eig};
	Contiguous1DMapInp<2> guess_map_inp( guess_shape );
	std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_guess_map = guess_map_inp.create_map();

    auto ptr_guess = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(ptr_comm, ptr_guess_map);
    auto ptr_guess2 = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(ptr_comm, ptr_guess_map);
    auto ptr_guess3 = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(ptr_comm, ptr_guess_map);
    auto ptr_guess4 = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(ptr_comm, ptr_guess_map);
    auto ptr_guess5 = std::make_unique< DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(ptr_comm, ptr_guess_map);
    // guess : unit vectors
    for(int i=0;i<num_eig;i++){
        std::array<int, 2> tmp_index = {i,i};
        ptr_guess->global_set_value(tmp_index, 1.0);
        ptr_guess2->global_set_value(tmp_index, 1.0);
        ptr_guess3->global_set_value(tmp_index, 1.0);
        ptr_guess4->global_set_value(tmp_index, 1.0);
        ptr_guess5->global_set_value(tmp_index, 1.0);
    }
    
	DecomposeOption option_evd;
    option_evd.algorithm_type = DecomposeMethod::Direct;

   
    std::cout << "========================\nDense matrix diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix2, ptr_guess.get(), option_evd);
    std::cout << "========================\nDense matrix diag done" << std::endl;    

    print_eigenvalues( "Eigenvalues", num_eig, out1.get()->real_eigvals.data(), out1.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "geev, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;

    SE::SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_sparse( ptr_comm, map2_inp.create_map(), N*3);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            std::array<int,2> index = {i,j};
            if(i == j)                   test_sparse.global_insert_value(index, 2.0*((double)i-(double)N)   - invh2*5.0/2.0);
            if(i == j +1 || i == j -1)   test_sparse.global_insert_value(index, invh2*4.0/3.0);
            if(i == j +2 || i == j -2)   test_sparse.global_insert_value(index, invh2*(-1.0)/12.0);
            if(i == j +3 || i == j -3)   test_sparse.global_insert_value(index, 0.3);
            if(i == j +4 || i == j -4)   test_sparse.global_insert_value(index, -0.1);
            if(i!=j && i%100 == 0 && j%100 == 0)  test_sparse.global_insert_value(index, 0.01);
        }
    }
    test_sparse.complete();
    std::cout << "matrix construction complete" << std::endl;

    DecomposeOption option("ISI.yaml");
    std::cout << "\n========\nDense Tensor Operation,  Davidson" << std::endl;
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
    auto out2 = decompose(test_matrix3, ptr_guess2.get(), option);
    print_eigenvalues( "Eigenvalues", num_eig, out2.get()->real_eigvals.data(), out2.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, Dense Matrix, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;

    std::cout << "\n========\nTest Tensor Operation,  Davidson" << std::endl;
    TestTensorOperations<MTYPE::Contiguous1D,DEVICETYPE::MKL> test_op(N);//= new TestTensorOperations<MTYPE::Contiguous1D, DEVICETYPE::MKL>(N);
    std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();  
    auto out4 = decompose(&test_op, ptr_guess3.get(), option);
    print_eigenvalues( "Eigenvalues", num_eig, out4.get()->real_eigvals.data(), out4.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, tensorOP, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end4 - begin4).count())/1000000.0 << "[sec]" << std::endl;

    std::cout << "\n========\nSparsematrix Davidson" << std::endl;
    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();  
    auto out3 = decompose(test_sparse, ptr_guess4.get(), option);
    print_eigenvalues( "Eigenvalues", num_eig, out3.get()->real_eigvals.data(), out3.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, Sparse, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count())/1000000.0 << "[sec]" << std::endl;

    std::cout << "\n========\n Python Tensor Operation,  Davidson" << std::endl;
    PyTensorOperations<MTYPE::Contiguous1D,DEVICETYPE::MKL> py_op("../example/tensor_operations.py");
    std::chrono::steady_clock::time_point begin5 = std::chrono::steady_clock::now();  
    auto out5 = decompose(&test_op, ptr_guess5.get(), option);
    print_eigenvalues( "Eigenvalues", num_eig, out5.get()->real_eigvals.data(), out5.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end5 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, PytensorOP, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end5 - begin5).count())/1000000.0 << "[sec]" << std::endl;

  return 0;
}
