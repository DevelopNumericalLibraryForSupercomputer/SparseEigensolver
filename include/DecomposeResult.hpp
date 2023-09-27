#pragma once
#include <array>
#include <iostream>

namespace TH{
template <typename datatype, size_t dimension, typename device>
class DecomposeResult{
public:
    //std::array<datatype*, dimension> factor_matrices;
    std::array<std::pair<size_t, size_t>, dimension> factor_matrix_sizes;
    //datatype* core_tensor;
};
/*
typedef enum{
    Real,
    RealSym,
    Complex,
    Hermitian,
} MAT_TYPE;
*/
// EigenDecomposeResult
// SysEig?
// Lanczos?

template <double, 2, typename device>
class EigenDecomposeResult: public DecomposeResult<double, 2, device>{
};



}
