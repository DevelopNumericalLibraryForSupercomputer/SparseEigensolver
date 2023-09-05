#include "Matrix.hpp"
#include <vector>
#include <iostream>

int main(){
    size_t m, k, n;
    m = 4;
    k = 2;
    n = 3;
    std::vector<double> A_vec = {1,2,3,4,5,6,7,8};
    std::vector<double> B_vec = {1,2,3,4,5,6};
    std::vector<double> AT_vec = {1,5,2,6,3,7,4,8};
    std::vector<double> BT_vec = {1,3,5,2,4,6};
    std::cout << m << " " << k << " " << n << std::endl;
    
    TensorHetero::Matrix<double> A = TensorHetero::Matrix<double>(m,k,A_vec);
    TensorHetero::Matrix<double> B = TensorHetero::Matrix<double>(k,n,B_vec);
    TensorHetero::Matrix<double> C = TensorHetero::Matrix<double>(m,n);

    TensorHetero::Matrix<double> AT = TensorHetero::Matrix<double>(k,m,AT_vec);
    TensorHetero::Matrix<double> BT = TensorHetero::Matrix<double>(n,k,BT_vec);
    

    C.multiply(A,B,"NoT","NoT");
    std::cout << A << B << C << std::endl;

    C.multiply(A,BT,"NoT","T");
    std::cout << A << BT << C << std::endl;

    C.multiply(AT,B,"T","NoT");
    std::cout << AT << B << C << std::endl;

    C.multiply(AT,BT,"T","T");
    std::cout << AT << BT << C << std::endl;


    return 0;
}
