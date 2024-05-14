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

//template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device, STORETYPE store> 
//std::unique_ptr<DecomposeResult<DATATYPE> > decompose(Tensor<dimension,DATATYPE,MAPTYPE,device,store>& tensor, std::string method);

template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(DenseTensor<2, DATATYPE, MAPTYPE, device>& tensor, DenseTensor<2, DATATYPE, MAPTYPE, device>* eigvec, std::string method)
{
    if(method == "evd"){
        return evd(tensor, eigvec);
    }
    else if(method == "davidson"){
        DenseTensorOperations* basic_op = new DenseTensorOperations(tensor);
        auto return_val = davidson(basic_op, eigvec);
        free(basic_op);
        return return_val;
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(SparseTensor<2, DATATYPE, MAPTYPE, device>& tensor, DenseTensor<2, DATATYPE, MAPTYPE, device>* eigvec, std::string method)
{
    if(method == "davidson"){
        SparseTensorOperations* basic_op = new SparseTensorOperations(tensor);
        auto return_val = davidson(basic_op, eigvec);
        free(basic_op);
        return return_val;
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(TensorOperations* operations, DenseTensor<2, DATATYPE, MAPTYPE, device>* eigvec, std::string method)
{
    if(method == "davidson"){
        return davidson(operations, eigvec);
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


}
