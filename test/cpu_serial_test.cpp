//#include "Matrix.hpp"
#include "Contiguous1DMap.hpp"
#include "DenseTensor.hpp"
#include "SparseTensor.hpp"
//#include "device/mkl/LinearOp.hpp"
#include "device/mkl/TensorOp.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <chrono>
//#include "device/MKL/TensorOp.hpp"
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
    auto ptr_comm = create_comm<DEVICETYPE::MKL>(argc, argv);
    std::cout << "SERIAL test" << std::endl;
    std::cout << *ptr_comm <<std::endl;    
    
    std::array<size_t, 2> test_shape = {4,3};
    Contiguous1DMap map (test_shape,  0,1);
    std::vector<double> test_data = {1,0,0,1,1,0,1,1,1,0,0,0};

    SE::DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> test_matrix(*ptr_comm,map);
    for(int i=0;i<test_shape[0] * test_shape[1];i++){
        test_matrix.local_insert_value(i,test_data[i]);
    }
    test_matrix.complete();
    std::cout << test_matrix <<std::endl; 
    std::cout << "QR ortho" << std::endl;
    auto QRortho(test_matrix);
    TensorOp::orthonormalize(QRortho,"qr");
    std::cout << QRortho << std::endl;
    std::cout << "general ortho" << std::endl;
    auto generalortho(test_matrix);
    TensorOp::orthonormalize(generalortho,"normal");
    std::cout << generalortho << std::endl;

    std::array<size_t, 1> vector_shape = {3};
    Contiguous1DMap map1d (vector_shape,  0,1);
    SE::DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MKL> test_vector(*ptr_comm,map1d);
    //std::cout << test_vector << std::endl;
    for(int i=0;i<3;i++){
        test_vector.local_insert_value(i,1.0);
    }
    std::cout << test_vector << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix, test_vector) <<std::endl;
//    auto out = decompose(test_matrix, "evd");
//    print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());
//    
    std::cout << "========================\nDense matrix davidson test" << std::endl;
    size_t N = 10000;
    std::array<size_t, 2> test_shape2 = {N,N};
    Contiguous1DMap map2 (test_shape2,  0,1);
    std::array<size_t, 1> test_shape2_vec = {N};
    Contiguous1DMap map2_vec (test_shape2_vec,  0,1);
    
    // local potnetial v(x) = 2.0*(i-N) + spacing 0.2, 3th order kinetic energy matrix
    double invh2 = 1/0.2/0.2;
    double* test_data2 = malloc<double, DEVICETYPE::MKL>(N*N);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            test_data2[i+j*N] = 0;
            if(i == j)                  test_data2[i+j*N] += 0.0  - invh2*5.0/2.0;//2.0*((double)i-(double)N) 
            if(i == j +1 || i == j -1)  test_data2[i+j*N] += invh2*4.0/3.0;
            if(i == j +2 || i == j -2)  test_data2[i+j*N] -= invh2*1.0/12.0;
            //if(i == j +3 || i == j -3)  test_data2[i+j*N] += 0.3;
            //if(i == j +4 || i == j -4)  test_data2[i+j*N] -= 0.1;
            //if( i%13 == 0 && j%13 == 0) test_data2[i+j*N] += 0.01;
            //if(i>=3 || j>=3) test_data2[i+j*N] = 0.0;
        }
    }
    DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> test_matrix2(*ptr_comm, map2, test_data2);

    //std::cout << test_matrix2.data[0] <<std::endl; 
    /*

    SE::DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> test_vec_long( *ptr_comm, map2_vec);
    test_vec_long.global_set_value(0,1.0);
    test_vec_long.global_set_value(3,1.0);
    std::cout <<  test_vec_long << std::endl;
    std::cout << "SPMV" << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix2, test_vec_long, TRANSTYPE::N ) <<std::endl;
    std::cout << "gemm" << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix2, test_matrix2 ) <<std::endl;
    
    std::cout << "========================\nDense matrix diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix2, "evd");
    print_eigenvalues( "Eigenvalues", 3, out1.get()->real_eigvals.get(), out1.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "geev, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    */
    
    SE::SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> test_sparse( *ptr_comm, map2, N*9);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(i == j)                   test_sparse.global_insert_value(index, 2.0*((double)i-(double)N) - invh2*5.0/2.0);
            if(i == j +1 || i == j -1)   test_sparse.global_insert_value(index, invh2*4.0/3.0);
            if(i == j +2 || i == j -2)   test_sparse.global_insert_value(index, invh2*(-1.0)/12.0);
            //if(i == j +3 || i == j -3)   test_sparse.global_insert_value(index, invh2*(-1.0)/12.0);
            //if(i == j +4 || i == j -4)   test_sparse.global_insert_value(index, -0.1);
            //if( i%13 == 0 && j%13 == 0)  test_sparse.global_insert_value(index, 0.01);
            //if( (j*N+i)%53 == 0) test_sparse.global_insert_value(index, 0.01);
        }
    }
    test_sparse.complete();
    std::cout << "matrix construction complete" << std::endl;
/*
    std::cout <<  test_sparse << std::endl;
    std::cout << "SPMV" << std::endl;
    std::cout <<  TensorOp::matmul( test_sparse, test_vec_long, TRANSTYPE::N ) <<std::endl;
    std::cout << "sgemm" << std::endl;
    std::cout <<  TensorOp::matmul( test_sparse, test_matrix2, TRANSTYPE::N, TRANSTYPE::N ) <<std::endl;
*/
    SE::DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MKL> test_matrix3(*ptr_comm, map2, test_data2 );
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
    auto out2 = decompose(test_matrix3, "davidson");
    print_eigenvalues( "Eigenvalues", 3, out2.get()->real_eigvals.get(), out2.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;
 
//
//    std::cout << "\nSparsematrix Davidson" << std::endl;
//    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();  
//    auto out3 = decompose(test_sparse, "davidson");
//    print_eigenvalues( "Eigenvalues", 3, out3.get()->real_eigvals.get(), out3.get()->imag_eigvals.get());
//    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
//    std::cout << "BlockDavidson_sparse, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count())/1000000.0 << "[sec]" << std::endl;
//
  return 0;
}
