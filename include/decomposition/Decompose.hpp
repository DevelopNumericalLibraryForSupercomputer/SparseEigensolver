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

//template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device, STORETYPE store> 
//std::unique_ptr<DecomposeResult<DATATYPE> > decompose(Tensor<dimension,DATATYPE,mtype,device,store>& tensor, std::string method);

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(DenseTensor<2, DATATYPE, mtype, device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec, std::string method)
{
    if(method == "evd"){
        return evd(tensor, eigvec);
    }
    else if(method == "davidson"){
        DenseTensorOperations<mtype, device>* basic_op = new DenseTensorOperations<mtype,device>(tensor);
        auto return_val = davidson(basic_op, eigvec);
        delete basic_op;
        return return_val;
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(SparseTensor<2, DATATYPE, mtype, device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec, std::string method)
{
    if(method == "davidson"){
        std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Sparse call " <<std::endl;
        SparseTensorOperations<mtype, device>* basic_op = new SparseTensorOperations(tensor);
        auto return_val = davidson(basic_op, eigvec);
        delete basic_op;
        return return_val;
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(TensorOperations<mtype,device>* operations, DenseTensor<2, DATATYPE, mtype, device>* eigvec, std::string method)
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
