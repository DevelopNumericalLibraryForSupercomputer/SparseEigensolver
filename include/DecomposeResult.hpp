#pragma once
#include <array>
#include <iostream>
#include "Comm_include.hpp"
namespace SE{
template <typename datatype, size_t dimension, typename comm>
class DecomposeResult{
public:
    //std::array<datatype*, dimension> factor_matrices;
    std::array<std::pair<size_t, size_t>, dimension> factor_matrix_sizes;
    //datatype* core_tensor;
};

typedef enum{
    Real,
    RealSym,
    Complex,
    Hermitian,
} MAT_TYPE;

// EigenDecomposeResult
// SysEig?
// Lanczos?
/*
template <typename datatype, size_t dimension, typename comm>
class EigenDecomposeResult: public DecomposeResult{
public:
    //std::array<std::pair<size_t, size_t>, dimension> factor_matrix_sizes;
};
*/


}
