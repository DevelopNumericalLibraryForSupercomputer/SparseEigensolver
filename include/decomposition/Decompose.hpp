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

//template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device, STORETYPE store> 
//std::unique_ptr<DecomposeResult<DATATYPE> > decompose(Tensor<dimension,DATATYPE,MAPTYPE,device,store>& tensor, std::string method);

template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(DenseTensor<2, DATATYPE, MAPTYPE, device>& tensor, std::string method)
{
    if(method == "evd"){
        return evd(tensor);
    }
    else if(method == "davidson"){
        BasicDenseTensorOperations basic_op(tensor);
        return davidson(basic_op);
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};

template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device, STORETYPE store> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(TensorOperations<2,DATATYPE,MAPTYPE,device,store>& operations, std::string method)
{
    if(method == "davidson"){
        return davidson(operations);
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


}
