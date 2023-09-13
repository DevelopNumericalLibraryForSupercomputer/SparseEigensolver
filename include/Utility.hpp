//ChatGPT3.5
#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "Device.hpp"

namespace TH{
template<typename datatype, typename device> datatype* malloc(const size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}
template<typename datatype, typename device> void free(datatype* ptr){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}
template<typename datatype, typename device> void memcpy(datatype* dest, const datatype* source, size_t size){
    std::cout <<  "This is not implemented yet" << std::endl;
    exit(-1);
}


//numerical recipies
template <size_t dimension>
void cumprod(const std::array<size_t, dimension>& shape, std::array<size_t, dimension+1>& shape_mult, std::string indexing="F"){
    /* Ex1)
     * shape = {2, 3, 4}, indexing="F"
     * shape_mult = {1, 2, 6, 24}
     * Ex2)
     * shape = {2, 3, 4}, indexing="C"
     * shape_mult = {1, 4, 12, 24}
     */
    shape_mult[0] = 1;
    if (indexing == "F"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[i];
        }
    }
    else if(indexing == "C"){
        for (size_t i = 0; i < dimension; ++i) {
            shape_mult[i+1] = shape_mult[i] * shape[dimension-i-1];
        }
    }
}

}