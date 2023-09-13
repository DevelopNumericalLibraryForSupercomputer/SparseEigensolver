//ChatGPT3.5
#pragma once
#include "Utility.hpp"

namespace TH{
template<>
double* malloc<double, CPU>(const size_t size) {
    return static_cast<double*>(std::malloc(size * sizeof(double)));
}
template<>
void free<double, CPU>(double* ptr) {
    std::free(ptr);
}
template<>
void memcpy<double, CPU>(double* dest, const double* source, size_t size){
    std::memcpy(dest, source, size * sizeof(double));
}

}