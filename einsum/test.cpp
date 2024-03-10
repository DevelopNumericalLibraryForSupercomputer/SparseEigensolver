#include "einsum.hpp"
#include <array>
#include <iostream>


int main() {
    const char* input_string = "ijl,jk->kli";
    double a[3*4*5] = 
    {1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 
0.0, 2.0, 0.0, 0.0, 0.0,  0.0, 0.0, 2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 2.0, 0.0,  0.0, 0.0, 0.0, 0.0, 2.0,
0.0, 0.0, 0.0, 0.0, 3.0,  0.0, 0.0, 0.0, 3.0, 0.0,  0.0, 0.0, 3.0, 0.0, 0.0,  0.0, 3.0, 0.0, 0.0, 0.0};

    double b[2*4] = {2,5, 0,1, 5,7, 9,2};
    
    std::array<size_t, 3> a_size = {3,4,5};
    std::array<size_t, 2> b_size = {4,2};

    std::unique_ptr<double[]> c;
    std::array<size_t, 3>* c_size = new std::array<size_t,3>;

/*
    const char* input_string = "ij,ij->ij";
    double a[2*2] = {1,2,3,4};
    double b[2*2] = {2,3,4,5};
    
    std::array<size_t, 2> a_size = {2,2};
    std::array<size_t, 2> b_size = {2,2};

    std::unique_ptr<double[]> c;
    std::array<size_t, 2>* c_size = new std::array<size_t,2>{2,2};
*/
    einsum(input_string, a, b, c, a_size, b_size, c_size);
    std::cout << "==================RESULT===============" << std::endl;
    for(int i=0;i<3;i++){
        std::cout << c_size->at(i) << ' ';
    }
    std::cout << std::endl;
    
    auto c_map = SE::Contiguous1DMap(*c_size, 0, 1);

    for(size_t i=0;i<c_size->at(0);++i){
        std::cout << "[";
        for(size_t j=0;j<c_size->at(1);++j){
            std::cout << "[";
            for(size_t k=0;k<c_size->at(2);++k){
                //std::cou
                std::array<size_t,3> index = {i,j,k};
                std::cout << c[c_map.unpack_global_array_index(index)] << ", ";
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]," << std::endl;
    }
    return 0;
}