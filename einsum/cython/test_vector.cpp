#include "einsum_vector.hpp"
#include <array>
#include <iostream>
#include <memory>

int main() {
    const char* input_string = "ijl,jk->kli";
    double a[3*4*5] = 
    {1.0, 0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 0.0, 0.0,  0.0, 0.0, 0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 0.0, 0.0, 
0.0, 2.0, 0.0, 0.0, 0.0,  0.0, 0.0, 2.0, 0.0, 0.0,  0.0, 0.0, 0.0, 2.0, 0.0,  0.0, 0.0, 0.0, 0.0, 2.0,
0.0, 0.0, 0.0, 0.0, 3.0,  0.0, 0.0, 0.0, 3.0, 0.0,  0.0, 0.0, 3.0, 0.0, 0.0,  0.0, 3.0, 0.0, 0.0, 0.0};

    double b[2*4] = {2,5, 0,1, 5,7, 9,2};
    /*
    std::array<size_t, 3> a_size = {3,4,5};
    std::array<size_t, 2> b_size = {4,2};
    */
    std::vector<size_t> a_size = {3,4,5};
    std::vector<size_t> b_size = {4,2};

    std::string result_index;
    std::vector<std::string> inputs_exprs;
    std::map<std::string, size_t> total_iter_sizes;
    input_parser(input_string, a_size, b_size, &result_index, &inputs_exprs, &total_iter_sizes);
    // c의 tensor size 추출 
    std::vector<size_t> c_size = calculate_return_size(result_index, total_iter_sizes);

    size_t c_whole_size = 1;
    for(size_t val : c_size){
        c_whole_size*=val;
    }

    std::unique_ptr<double[]> c = std::make_unique<double[]>(c_whole_size);
    einsum(input_string, a, b, c.get(), a_size, b_size);

    std::cout << "==================RESULT===============" << std::endl;
    for(int i=0;i<3;i++){
        std::cout << c_size.at(i) << ' ';
    }
    std::cout << std::endl;
    
    //auto c_map = SE::Contiguous1DMap(*c_size, 0, 1);

    for(size_t i=0;i<c_size.at(0);++i){
        std::cout << "[";
        for(size_t j=0;j<c_size.at(1);++j){
            std::cout << "[";
            for(size_t k=0;k<c_size.at(2);++k){
                //std::cou
                std::vector<size_t> index = {i,j,k};
                std::cout << c[unpack_array_index(index, c_size)] << ", ";
            }
            std::cout << "]," << std::endl;
        }
        std::cout << "]," << std::endl;
    }
    return 0;
}