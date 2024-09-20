#include <array>
#include <vector>
#include <iostream>

#include "device/mkl/TensorOp.hpp"
#include "Contiguous1DMap.hpp"
#include "DenseTensor.hpp"
#include "SparseTensor.hpp"

using namespace SE;

int main(int argc, char* argv[]){
    //Initialize MKL Comm
    std::cout << "# Create Comm" << std::endl;
	MKLCommInp comm_inp;
    auto ptr_comm = comm_inp.create_comm();
    
    //Initialize Contiguous1DMap
    //4 by 3 matrix
    std::cout << "# Create Map" << std::endl;
    std::array<int, 2> test_shape = {4,3};
	Contiguous1DMapInp<2> map_inp( test_shape );
	std::unique_ptr<Map<2,MTYPE::Contiguous1D> > ptr_map = map_inp.create_map();
    //Map information
	std::cout << *ptr_map <<std::endl;

    //Initialize DenseTensor (4 by 3 matrix)
    //  1.00  0.00  0.00 
    //  1.00  1.00  0.00 
    //  1.00  1.00  1.00 
    //  0.00  0.00  0.00 
    std::cout << "# Create DenseTensor (2D matrix)" << std::endl;
    std::cout << "3 column vectors with 4 entities" << std::endl;
    std::vector<double> test_data = {1,0,0,1,1,0,1,1,1,0,0,0};
    SE::DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix(ptr_comm,ptr_map);
    for(int i=0;i<test_shape[0] * test_shape[1];i++){
        test_matrix.local_insert_value(i,test_data[i]);
    }
    test_matrix.complete();
    std::cout << test_matrix <<std::endl; 
    
    //Lowdin Symmetric Orthogonalization (default)
    std::cout << "# Symmetric orthogonalization" << std::endl;
    //copy test_matrix to Symortho
    auto Symortho(test_matrix);
    
    using TensorOp = TensorOp<MTYPE::Contiguous1D,DEVICETYPE::MKL>;

    TensorOp::orthonormalize(Symortho,"symmetric");
    std::cout << Symortho << std::endl;

    //QR decomposition
    std::cout << "# QR orthonormalization" << std::endl;
    //copy test_matrix to QRmatrix
    auto QRortho(test_matrix);
    TensorOp::orthonormalize(QRortho,"qr");
    std::cout << QRortho << std::endl;

    //Initialize DenseTensor (3 by 1 vector)
    // {1.0, 1.0, 1.0}
    std::cout << "# Create DenseTensor (1D matrix)" << std::endl;
    std::cout << "1 column vectors with 3 entities" << std::endl;
    std::array<int, 1> vector_shape = {3};
	Contiguous1DMapInp<1> map1d_inp( vector_shape );
	std::unique_ptr<Map<1,MTYPE::Contiguous1D> > ptr_map1d = map1d_inp.create_map();
    SE::DenseTensor<1,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_vector(ptr_comm,ptr_map1d);
    for(int i=0;i<3;i++){
        test_vector.local_insert_value(i,1.0);
    }
    std::cout << test_vector << std::endl;
    
    //Matrix-Vector Multiplication (dense-dense)
    std::cout << "# Matrix-matrix Multiplication (dense-dense)" << std::endl;
    std::cout <<  *(TensorOp::matmul( test_matrix, test_vector)) <<std::endl;
    
    std::cout << "##############################\n";
    std::cout << "# Sparse matrix operation" << std::endl;
    
    // create DenseTensor from external data (N by N matrix)
    // Up to now, data should have form of std::unique_ptr<double[], std::function<void(double*)> >
    std::cout << "# DenseTensor (N by N matrix), Fourth-order finite difference operator matrix for the second derivative" << std::endl;
    int N = 20;
    std::array<int, 2> test_shape2 = {N,N};
	Contiguous1DMapInp<2> map2_inp( test_shape2 );
    std::unique_ptr<double[], std::function<void(double*)> > test_data2 ( malloc<double, DEVICETYPE::MKL>(N*N), free<DEVICETYPE::MKL> );
    // 3th order kinetic energy matrix
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            test_data2.get()[i+j*N] = 0;
            if(i == j)                  test_data2.get()[i+j*N] -= 5.0/2.0;
            if(i == j +1 || i == j -1)  test_data2.get()[i+j*N] += 4.0/3.0;
            if(i == j +2 || i == j -2)  test_data2[i+j*N] -= 1.0/12.0;
        }
    }
    DenseTensor<2,double,MTYPE::Contiguous1D, DEVICETYPE::MKL> test_matrix2(ptr_comm, map2_inp.create_map(), std::move(test_data2));
    std::cout << test_matrix2 << std::endl; 

    std::cout << "# SparseTensor (N by N matrix), Fourth-order finite difference operator matrix for the second derivative" << std::endl;
    SE::SparseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_sparse( ptr_comm, map2_inp.create_map(), N*3);
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            std::array<int,2> index = {i,j};
            if(i == j)                   test_sparse.global_insert_value(index, - 5.0/2.0);
            if(i == j +1 || i == j -1)   test_sparse.global_insert_value(index, 4.0/3.0);
            if(i == j +2 || i == j -2)   test_sparse.global_insert_value(index, (-1.0)/12.0);
        }
    }
    test_sparse.complete();
    std::cout << test_sparse << std::endl; 

    // DenseTensor (one vector with N entities)
    std::cout << "# DenseTensor (1 by N matrix)" << std::endl;
    std::array<int, 1> test_shape2_vec = {N};
	Contiguous1DMapInp<1> map2_vec_inp( test_shape2_vec );
	std::unique_ptr<Map<1,MTYPE::Contiguous1D> > ptr_map2_vec = map2_vec_inp.create_map();
    SE::DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> test_vec_long(ptr_comm, map2_vec_inp.create_map());
    test_vec_long.global_set_value(0,1.0);
    test_vec_long.global_set_value(3,1.0);
    std::cout <<  test_vec_long << std::endl;       

    std::cout << "# Sparse matrix - vector multiplication" << std::endl;
    std::cout <<  *(TensorOp::matmul( test_sparse, test_vec_long, TRANSTYPE::N) ) <<std::endl;
    std::cout << "# Sparse matrix - dense matrix multiplication" << std::endl;
    std::cout <<  *(TensorOp::matmul( test_sparse, test_matrix2, TRANSTYPE::N, TRANSTYPE::N ) ) <<std::endl;
    
  return 0;
}
