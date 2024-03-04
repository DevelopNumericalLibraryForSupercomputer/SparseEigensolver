#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "../include/Contiguous1DMap.hpp"

// row major
template <size_t A_dim, size_t B_dim, size_t C_dim>
void einsum(const char* input_string, const double* a, const double* b, std::unique_ptr<double[]>& c, 
            const std::array<size_t, A_dim> a_size, const std::array<size_t, B_dim> b_size, std::array<size_t, C_dim>* c_size) {

    //input_string에서 index 번호 추출, input tensor 2개 output tensor 1개
    std::string input_str(input_string); // 문자열 리터럴을 std::string으로 변환
    size_t arrow_pos = input_str.find("->");

    std::string result_index;
    std::vector<std::string> inputs_exprs;

    if (arrow_pos != std::string::npos) {
        result_index = input_str.substr(arrow_pos + 2);
        std::string input_indices = input_str.substr(0, arrow_pos);

        size_t comma_pos = 0;
        while ((comma_pos = input_indices.find(",")) != std::string::npos) {
            inputs_exprs.push_back(input_indices.substr(0, comma_pos));
            input_indices.erase(0, comma_pos + 1);
        }
        inputs_exprs.push_back(input_indices); // 마지막 토큰 추가

    } else {
        std::cerr << "Arrow (->) not found in the input string." << std::endl;
    }

    // 결과 출력
    std::cout << "Result Index: " << result_index << std::endl;
    std::cout << "Input Indices: ";
    for (const std::string& index : inputs_exprs) {
        std::cout << index << " ";
    }
    std::cout << std::endl;
    
    //각 index letter별 tensor size 저장.
    std::map<std::string, size_t> total_iter_sizes;
    //sizes stores size of each index, using key as single letter string.
    // For example, i: 5, j: 4, and so on.
    for (size_t j = 0; j < inputs_exprs[0].size(); ++j) {
        std::string key(1, inputs_exprs[0][j]);
        size_t size = a_size[j];
        total_iter_sizes[key] = size;
    }
    for (size_t j = 0; j < inputs_exprs[1].size(); ++j) {
        std::string key(1, inputs_exprs[1][j]);
        size_t size = b_size[j];
        total_iter_sizes[key] = size;
    }
    /* c의 tensor size 추출 */
    //delete c_size;
    assert(result_index.size() == C_dim);
    //c_size = new std::array<size_t, C_dim>();
    int tmp_index = 0;
    for (char key : result_index) {
        std::string key_str(1, key);  // 문자를 문자열로 변환
        auto it = total_iter_sizes.find(key_str);
        std::cout << "C : " << tmp_index << " : "; 
        if (it != total_iter_sizes.end()) {
            c_size->at(tmp_index) = it->second;
            std::cout << c_size->at(tmp_index)  << std::endl;
            tmp_index++;
        }
        
        
    }
    
    //c 초기화
    size_t c_total_size = 1;
    for (size_t size : *c_size) {
        c_total_size *= size;
        //std::cout << size << ' ';
    }
    //std::cout << '\n' <<  c_total_size << std::endl;
    //c = static_cast<double*>(std::malloc(sizeof(double)*c_total_size));
    c.reset(new double[c_total_size]{});
    //std::memset(c,0.0,sizeof(double)*c_total_size);

    //각 tensor의 pack/unpack 함수를 위해 Map 사용. 
    auto a_map = SE::Contiguous1DMap(a_size, 0, 1);
    auto b_map = SE::Contiguous1DMap(b_size, 0, 1);
    auto c_map = SE::Contiguous1DMap(*c_size, 0, 1);

    //모든 index에 대해 for 문을 돌아야함.

    //total number of iteration 계산
    size_t total_combinations = 1;

    std::map<std::string, size_t> indices;

    for (const auto& kv : total_iter_sizes) {        
        indices[kv.first] = 0;
        total_combinations *= kv.second;
        // 결과 출력
        //std::cout << "Key: " << kv.first << ", Size: " << kv.second << std::endl;
    }
    //one large loop
    for (size_t combination = 0; combination < total_combinations; ++combination) {
        //검증
        /*
        for (const auto& kv : indices){
            std::cout << kv.first << " : " << kv.second << " ";
        }
        std::cout << '\t';
        */
        //tensor index 확인
        std::array<size_t, A_dim> a_index;
        //std::cout << "a : (";
        for (size_t j = 0; j < inputs_exprs[0].size(); ++j) {
            std::string key(1, inputs_exprs[0][j]);
            a_index[j] = indices[key];
            //std::cout << key << " : " << a_index[j] << ' ';
        }
        //std::cout << ")\t";

        std::array<size_t, B_dim> b_index;
        //std::cout << "b : (";
        for (size_t j = 0; j < inputs_exprs[1].size(); ++j) {
            std::string key(1, inputs_exprs[1][j]);
            b_index[j] = indices[key];
            //std::cout << key << " : " << b_index[j] << ' ';
        }
        //std::cout << ")\t";
        
        std::array<size_t, C_dim> c_index;
        //std::cout << "c : (";
        for (size_t j = 0; j < result_index.size(); ++j) {
            std::string key(1, result_index[j]);
            c_index[j] = indices[key];
            //std::cout << key << " : " << c_index[j] << ' ';
        }
        //std::cout << ")" << std::endl;

        c[c_map.unpack_global_array_index(c_index)] += a[a_map.unpack_global_array_index(a_index)] * b[b_map.unpack_global_array_index(b_index)];

        //c = a * b

        //index 변경 000, 100, 200, ..., 010, 110, ...
        for (auto it = indices.begin(); it != indices.end(); ++it){
            it->second += 1;
            if (it->second < total_iter_sizes[it->first]){
                break;
            }
            it->second = 0;
        }
    }

}



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