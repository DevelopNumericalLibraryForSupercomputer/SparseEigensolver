#pragma once
#include <memory>
#include <functional>
#include "DecomposeResult.hpp"
#include "DirectSolver.hpp"
#include "IterativeSolver.hpp"

namespace SE{

/*
template<typename DATATYPE, typename comm, typename map> 
std::unique_ptr<DecomposeResult<DATATYPE, 2, comm, map> > decompose(std::function<DenseTensor<DATATYPE,2,comm,map> (DenseTensor<DATATYPE,2,comm,map>) >& matvec, std::string method)
{static_assert(false, "not implemented yet"); };
*/
//std::unique_ptr<DecomposeResult<DATATYPE, 2, comm, map> > decompose(DenseTensor<DATATYPE,2,comm,map>& tensor, std::string method)

template<STORETYPE storetype, typename DATATYPE, size_t dimension, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(Tensor<storetype, DATATYPE, dimension, device, MAPTYPE>& tensor, std::string method)
{static_assert(false, "not implemented yet"); };

template<STORETYPE storetype, typename DATATYPE, DEVICETYPE device, typename MAPTYPE>
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(Tensor<storetype, DATATYPE, 2, device, MAPTYPE>& tensor, std::string method)
{
    if(method == "evd"){
        return evd(tensor);
    }
    else if(method == "davidson"){
        return davidson(tensor);
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};

}
