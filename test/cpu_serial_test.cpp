//#include "Matrix.hpp"
//#include "device/mkl/LinearOp.hpp"

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

#include "decomposition/TestOperations.hpp"
#include "decomposition/Decompose.hpp"

std::ostream& operator<<(std::ostream& os, std::array<int,3> &A){
    os << "(";
    for(int i = 0; i<3; i++){
        os << A[i] << " ";
    }
    os << ")";
    return os;
}

using namespace SE;

int main(int argc, char* argv[]){
	MKLCommInp comm_inp;
    auto ptr_comm = comm_inp.create_comm();
    std::cout << "SERIAL test" << std::endl;
    std::cout << *ptr_comm <<std::endl;    
    /*
    std::array<int, 2> test_shape = {4,3};
	Contiguous1DMapInp<2> map_inp( test_shape );
	std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_map = map_inp.create_map();
	std::cout << *ptr_map <<std::endl;

    std::vector<double> test_data = {1,0,0,1,1,0,1,1,1,0,0,0};
    SE::DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix(ptr_comm,ptr_map);
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

    std::array<int, 1> vector_shape = {3};
	Contiguous1DMapInp<1> map1d_inp( vector_shape );
	std::unique_ptr<Map<1,MTYPE::Contiguous1D> > ptr_map1d = map1d_inp.create_map();
    SE::DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_vector(ptr_comm,ptr_map1d);
    //std::cout << test_vector << std::endl;
    for(int i=0;i<3;i++){
        test_vector.local_insert_value(i,1.0);
    }
    std::cout << test_vector << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix, test_vector) <<std::endl;
    */
   
    std::cout << "========================\nDense matrix davidson test" << std::endl;
    int N = 10;
    const int num_eig = 3;


    std::array<int, 2> test_shape2 = {N,N};
	Contiguous1DMapInp<2> map2_inp( test_shape2 );
    
    std::array<int, 1> test_shape2_vec = {N};
	Contiguous1DMapInp<1> map2_vec_inp( test_shape2_vec );
	std::unique_ptr<Map<1,MTYPE::Contiguous1D> > ptr_map2_vec = map2_vec_inp.create_map();
    
    // local potnetial v(x) = 2.0*(std::abs(0.5*(double)N-(double)i)) + spacing 1.0, 3th order kinetic energy matrix
    double invh2 = 1.0;
    std::unique_ptr<double[], std::function<void(double*)> > test_data2 ( malloc<double, DEVICETYPE::MKL>(N*N), free<DEVICETYPE::MKL> );
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            test_data2.get()[i+j*N] = 0;
            if(i == j)                  test_data2.get()[i+j*N] += 2.0*((double)i-(double)N)   - invh2*5.0/2.0;//2.0*((double)i-(double)N) 
            if(i == j +1 || i == j -1)  test_data2.get()[i+j*N] += invh2*4.0/3.0;
            if(i == j +2 || i == j -2)  test_data2[i+j*N] -= invh2*1.0/12.0;
            if(i == j +3 || i == j -3)  test_data2[i+j*N] += 0.3;
            //if(i == j +4 || i == j -4)  test_data2[i+j*N] -= 0.1;
            //if( i%13 == 0 && j%13 == 0) test_data2[i+j*N] += 0.01;
            //if(i>=3 || j>=3) test_data2[i+j*N] = 0.0;
        }
    }
    DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix2(ptr_comm, map2_inp.create_map(), std::move(test_data2));
    //DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix3(test_matrix2);
    //test_matrix2.complete();
    std::cout << test_matrix2 <<std::endl; 
    //std::cout << test_matrix3 <<std::endl; 

    /*
    
    SE::DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_vec_long(ptr_comm, map2_vec_inp.create_map());
    test_vec_long.global_set_value(0,1.0);
    test_vec_long.global_set_value(3,1.0);
    std::cout <<  test_vec_long << std::endl;
    std::cout << "SPMV" << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix2, test_vec_long, TRANSTYPE::N ) <<std::endl;
    std::cout << "gemm" << std::endl;
    std::cout <<  TensorOp::matmul( test_matrix2, test_matrix2 ) <<std::endl;
    
    */
   
    std::array<int, 2> guess_shape = {N,num_eig};
	Contiguous1DMapInp<2> guess_map_inp( guess_shape );
	std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_guess_map = guess_map_inp.create_map();
    
    
    DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>* guess = new DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(ptr_comm, ptr_guess_map);
    
    
    
    // guess : unit vector
    for(int i=0;i<num_eig;i++){
        std::array<int, 2> tmp_index = {i,i};
        guess->global_set_value(tmp_index, 1.0);
    }

    
    std::cout << "========================\nDense matrix diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix2, guess, "evd");
    std::cout << "========================\nDense matrix diag done" << std::endl;
    delete guess;
    
    
    print_eigenvalues( "Eigenvalues", num_eig, out1.get()->real_eigvals.data(), out1.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "geev, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    
    free<DEVICETYPE::MKL>(guess);
    
    /*
    guess = new DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>( ptr_comm, ptr_guess_map);
    // guess : unit vector
    for(int i=0;i<num_eig;i++){
        std::array<int, 2> tmp_index = {i,i};
        guess->global_set_value(tmp_index, 1.0);
    }
    
    SE::SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_sparse( ptr_comm, ptr_map2, N*9);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            std::array<int,2> index = {i,j};
            if(i == j)                   test_sparse.global_insert_value(index, 2.0*((double)i-(double)N)   - invh2*5.0/2.0);
            if(i == j +1 || i == j -1)   test_sparse.global_insert_value(index, invh2*4.0/3.0);
            //if(i == j +2 || i == j -2)   test_sparse.global_insert_value(index, invh2*(-1.0)/12.0);
            //if(i == j +3 || i == j -3)   test_sparse.global_insert_value(index, invh2*(-1.0)/12.0);
            //if(i == j +4 || i == j -4)   test_sparse.global_insert_value(index, -0.1);
            //if( i%13 == 0 && j%13 == 0)  test_sparse.global_insert_value(index, 0.01);
            //if( (j*N+i)%53 == 0) test_sparse.global_insert_value(index, 0.01);
        }
    }
    test_sparse.complete();
    std::cout << "matrix construction complete" << std::endl;
    다시짜야함
    */
/*
    std::cout <<  test_sparse << std::endl;
    std::cout << "SPMV" << std::endl;
    std::cout <<  TensorOp::matmul( test_sparse, test_vec_long, TRANSTYPE::N ) <<std::endl;
    std::cout << "sgemm" << std::endl;
    std::cout <<  TensorOp::matmul( test_sparse, test_matrix2, TRANSTYPE::N, TRANSTYPE::N ) <<std::endl;
*/
/*다시
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
    auto out2 = decompose(test_matrix3, guess, "davidson");
    print_eigenvalues( "Eigenvalues", num_eig, out2.get()->real_eigvals.data(), out2.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;

    free(guess);
    guess = new DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(ptr_comm, ptr_guess_map);
    // guess : unit vector
    for(int i=0;i<num_eig;i++){
        std::array<int, 2> tmp_index = {i,i};
        guess->global_set_value(tmp_index, 1.0);
    }

    TestTensorOperations<MTYPE::Contiguous1D,DEVICETYPE::MKL>* test_op = new TestTensorOperations<MTYPE::Contiguous1D, DEVICETYPE::MKL>(N);
    std::chrono::steady_clock::time_point begin4 = std::chrono::steady_clock::now();  
    auto out4 = decompose(test_op, guess, "davidson");
    print_eigenvalues( "Eigenvalues", num_eig, out4.get()->real_eigvals.data(), out4.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end4 = std::chrono::steady_clock::now();
    std::cout << "BlockDavidson, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end4 - begin4).count())/1000000.0 << "[sec]" << std::endl;
 다시짜야함
*/

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
