//ChatGPT3.5
#pragma once
#include <iostream>
#include <cstdlib>
#include "Device.hpp"

namespace TH{
//memory allocation
// Generic malloc and free functions (default)
template<typename datatype, typename device>
datatype* malloc(const size_t size) { return new datatype[size]; }

template<typename datatype, typename device>
void free(datatype* ptr) {   delete[] ptr; }

// Specialization for specific data types and devices
template<>
double* malloc<double, CPU>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}
template<>
void free<double, CPU>(double* ptr) {
    std::free(ptr);
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