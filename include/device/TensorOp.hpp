#pragma once
#include <string>
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"

namespace SE{
//spmv
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, 1, computEnv, maptype>* spmv(SparseTensor<datatype, dimension, computEnv, maptype> a, DenseTensor<datatype, 1, computEnv, maptype> v, SE_transpose transa = SE_transpose::NoTrans)
{  static_assert(false,"This is not implemented yet");  }    


//matmul
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension, computEnv, maptype>* matmul(DenseTensor<datatype, dimension, computEnv, maptype> a, DenseTensor<datatype, dimension, computEnv, maptype> b, SE_transpose transa, SE_transpose transb)
{  static_assert(false,"This is not implemented yet");  }    

//QR

template <typename datatype, typename computEnv>
void orthonormalize(datatype* eigvec, size_t vector_size, size_t number_of_vectors, std::string method)
{  static_assert(false,"This is not implemented yet");  }


}
