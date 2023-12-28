#pragma once
#include "../LinearOp.hpp"
#include "../../Utility.hpp"
#include <cublas_v2.h>

namespace SE
{
cublasOperation_t map_transtype(SE::TRANSTYPE trans){
    switch (trans){
        case SE::TRANSTYPE::N:     return CUBLAS_OP_N;
        case SE::TRANSTYPE::T:     return CUBLAS_OP_T;
        case SE::TRANSTYPE::C:     return CUBLAS_OP_C;
        default: throw std::runtime_error("map_transtype in device/CUDA/Utility.hpp");
    }

    // to supress warning;
    return CUBLAS_OP_N;
}

cudaMemcpyKind map_copytype(SE::COPYTYPE copy_type){
    switch (copy_type){
        case SE::COPYTYPE::NONE:          return cudaMemcpyHostToHost;
        case SE::COPYTYPE::HOST2DEVICE:   return cudaMemcpyHostToDevice;
        case SE::COPYTYPE::DEVICE2HOST:   return cudaMemcpyDeviceToHost;
        case SE::COPYTYPE::DEVICE2DEVICE: return cudaMemcpyDeviceToDevice;
        default: throw std::runtime_error("map_copytype in device/CUDA/Utility.hpp");
    }
    return  cudaMemcpyHostToHost;
}
}

