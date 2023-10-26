#pragma once
#include <memory>
#include <functional>
#include "../DenseTensor.hpp"
#include "../device/MKL/Utility.hpp"

namespace SE{

template<DecomposeMethod method, typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult<datatype, 2, comm, map>> decompose(std::function<DenseTensor<datatype,2,comm,map> (DenseTensor<datatype,2,comm,map>) >& matvec ){static_assert(false, "not implemented yet") };

template<DecomposeMethod method, typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult<datatype, 2, comm, map>> decompose(DenseTensor<datatype,2,comm,map>& tensor){static_assert(false, "not implemented yet") };


}
