//#include "Matrix.hpp"
//#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>
#include "ContiguousMap.hpp"
#include "device/MKL/LinearOp.hpp"
#include "device/MKL/TensorOp.hpp"
#include "decomposition/Decompose.hpp"

std::ostream& operator<<(std::ostream& os, std::array<size_t,3> &A){
    os << "(";
    for(int i = 0; i<3; i++){
        os << A[i] << " ";
    }
    os << ")";
    return os;
}

using namespace SE;

int main(int argc, char* argv[]){
    auto comm = createComm<MKL>(argc, argv);
    std::cout << "SERIAL test" << std::endl;
    std::cout << *comm <<std::endl;    
    
    std::array<size_t, 2> test_shape = {3,3};
    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    ContiguousMap<2>* new_map = new ContiguousMap(test_shape, 1);
    SE::Tensor<STORETYPE::Dense, double, 2, MKL  > test_matrix(comm.get(), new_map, test_shape, &test_data[0]);
    //test_matrix.print_tensor();
    auto out = decompose(test_matrix, "evd");
    print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());
    
    std::cout << "========================\nDense matrix davidson test" << std::endl;
    size_t N = 30;
    std::array<size_t, 2> test_shape2 = {N,N};
    ContiguousMap<2>* new_map2 = new ContiguousMap(test_shape2, 1);
    double* test_data2 = malloc<double, MKL>(N*N);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            test_data2[i+j*N] = 0;
            if(i == j)                  test_data2[i+j*N] += 2.0*((double)i+1.0-(double)N);
            //if(i == j +1 || i == j -1)  test_data2[i+j*N] += 3.0;
            if(i == j +2 || i == j -2)  test_data2[i+j*N] -= 1.0;
            if(i == j +3 || i == j -3)  test_data2[i+j*N] += 0.3;
            //if(i == j +4 || i == j -4)  test_data2[i+j*N] -= 0.1;
            //if( i%13 == 0 && j%13 == 0) test_data2[i+j*N] += 0.01;
        }
    }
    SE::Tensor<STORETYPE::Dense, double, 2, MKL, ContiguousMap<2> > test_matrix2(comm.get(), new_map2, test_shape2, test_data2);
    std::cout << "========================\nDense matrix davidson diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix2, "evd");
    print_eigenvalues( "Eigenvalues", 3, out1.get()->real_eigvals.get(), out1.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "geev, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    
    
    ContiguousMap<2>* new_map3 = new ContiguousMap(test_shape2, 1);
    SE::Tensor<STORETYPE::COO, double, 2, MKL, ContiguousMap<2> > test_sparse(comm.get(), new_map3, test_shape2, N*9);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(i == j)                   test_sparse.insert_value(index, 2.0*((double)i+1.0-(double)N) );
            //if(i == j +1 || i == j -1)   test_sparse.insert_value(index, 3.0);
            if(i == j +2 || i == j -2)   test_sparse.insert_value(index, -1.0);
            if(i == j +3 || i == j -3)   test_sparse.insert_value(index, 0.3);
            //if(i == j +4 || i == j -4)   test_sparse.insert_value(index, -0.1);
            //if( i%13 == 0 && j%13 == 0)  test_sparse.insert_value(index, 0.01);
            //if( (j*N+i)%53 == 0) test_sparse.insert_value(index, 0.01);
        }
    }
    test_sparse.complete();
    std::cout << "matrix construction complete" << std::endl;
    test_sparse.print();

    SE::Tensor<STORETYPE::Dense, double, 2, MKL, ContiguousMap<2> > test_matrix3(comm.get(), new_map2, test_shape2, test_data2);
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
    auto out2 = decompose(test_matrix3, "davidson");
    print_eigenvalues( "Eigenvalues", 3, out2.get()->real_eigvals.get(), out2.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;
 

    std::cout << "\nSparsematrix Davidson" << std::endl;
    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();  
    auto out3 = decompose(test_sparse, "davidson");
    print_eigenvalues( "Eigenvalues", 3, out3.get()->real_eigvals.get(), out3.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson_sparse, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count())/1000000.0 << "[sec]" << std::endl;

    return 0;
}
