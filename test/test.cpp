#include "Matrix.hpp"
#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include <iostream>
#include "Comm.hpp"
#include "Device.hpp"

int main(){
    TensorHetero::Device device;
    TensorHetero::Comm<double, TensorHetero::Device> comm;

    std::cout << "Matrix muliplication test" << std::endl;
    size_t m, k, n;
    m = 4;
    k = 2;
    n = 3;
    std::vector<double> A_vec = {1,2,3,4,5,6,7,8};
    std::vector<double> B_vec = {1,2,3,4,5,6};
    std::vector<double> AT_vec = {1,5,2,6,3,7,4,8};
    std::vector<double> BT_vec = {1,3,5,2,4,6};
    std::cout << m << " " << k << " " << n << std::endl;
    
    TensorHetero::Matrix<double> Amat = TensorHetero::Matrix<double>(m,k,A_vec);
    TensorHetero::Matrix<double> Bmat = TensorHetero::Matrix<double>(k,n,B_vec);
    TensorHetero::Matrix<double> Cmat = TensorHetero::Matrix<double>(m,n);

    TensorHetero::Matrix<double> AmatT = TensorHetero::Matrix<double>(k,m,AT_vec);
    TensorHetero::Matrix<double> BmatT = TensorHetero::Matrix<double>(n,k,BT_vec);
    
    
    Cmat.multiply(Amat,Bmat,"NoT","NoT");
    std::cout << Amat << Bmat << Cmat << std::endl;

    Cmat.multiply(Amat,BmatT,"NoT","T");
    std::cout << Amat << BmatT << Cmat << std::endl;

    Cmat.multiply(AmatT,Bmat,"T","NoT");
    std::cout << AmatT << Bmat << Cmat << std::endl;

    Cmat.multiply(AmatT,BmatT,"T","T");
    std::cout << AmatT << BmatT << Cmat << std::endl;

    std::cout << "DenseTensor Operator test" << std::endl;
    std::array<size_t,4> shape = {1,3,2,4};
    TensorHetero::DenseTensor<double,4,TensorHetero::Device,TensorHetero::Comm<double, TensorHetero::Device>> Aten = TensorHetero::DenseTensor<double,4,TensorHetero::Device,TensorHetero::Comm<double, TensorHetero::Device>>(shape);

    std::vector<double> Bten_vec = {};

    std::cout << Aten.data.size() << " : ";
    for(size_t i=0; i<Aten.data.size(); ++i){
        Aten[i]=1.0;
        std::cout << Aten[i] << " ";
        Bten_vec.push_back((double)i);
    }
    std::cout << std::endl;
    std::cout << "shape = (" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "), shape_mult : ";

    for(int i=0;i<Aten.shape_mult.size();++i){
        std::cout << Aten.shape_mult[i] << " ";
    }
    std::cout << std::endl;
    TensorHetero::DenseTensor<double,4,TensorHetero::Device,TensorHetero::Comm<double, TensorHetero::Device>> Bten = TensorHetero::DenseTensor<double,4,TensorHetero::Device,TensorHetero::Comm<double, TensorHetero::Device>>(shape,Bten_vec);
    std::array<size_t,4> index_array = {0,1,1,2};
    std::cout << "0,1,1,2 : " << Bten(index_array) << std::endl;
    index_array[3]=0;
    std::cout << "0,1,1,0 : " << Bten(index_array) << std::endl;
    index_array[1]=0;
    Bten.insert_value(index_array,-3.2);
    std::cout << "0,0,1,0 : " << Bten(index_array) << std::endl;
    Bten[8]=3.14;
    for(size_t i=0; i<Bten.data.size(); ++i){
        std::cout << Bten[i] << " " ;
    }
    std::cout << std::endl;
    TensorHetero::DenseTensor<double,4,TensorHetero::Device,TensorHetero::Comm<double, TensorHetero::Device>> Cten = Bten.clone();
    for(size_t i=0; i<Cten.data.size(); ++i){
        std::cout << Cten[i] << " " ;
    }
    std::cout << std::endl;
    Cten = Aten;
    for(size_t i=0; i<Cten.data.size(); ++i){
        std::cout << Cten[i] << " " ;
    }   


    return 0;
}
