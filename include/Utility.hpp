//ChatGPT3.5
#pragma once
#include <iostream>
#include <cstdlib>

namespace TensorHetero {
//memory allocation
// Generic malloc and free functions (default)
template<typename datatype, typename device>
datatype* malloc(const size_t size) { return new datatype[size]; }
template<typename datatype, typename device>
void free(datatype* ptr) {   delete[] ptr; }

/*
// Specialization for specific data types and devices
template<>
float* malloc<float, CPU>(const size_t size) {
    // Implement CPU-specific allocation for float data type
    // Example: Allocate memory on CPU
    return static_cast<float*>(std::malloc(size * sizeof(float)));
}
template<>
void free<float, CPU>(float* ptr) {
    // Implement CPU-specific deallocation for float data type
    // Example: Free memory on CPU
    std::free(ptr);
}
*/
//numerical recipies
template <size_t dimension>
void cumprod(const std::array<size_t, dimension>& shape, std::array<size_t, dimension+1>& shape_mult, std::string indexing="F");

template <size_t dimension>
void cumprod(const std::array<size_t, dimension>& shape, std::array<size_t, dimension+1>& shape_mult, std::string indexing){
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