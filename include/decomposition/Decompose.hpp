#pragma once
#include <memory>
#include <functional>
#include "DecomposeResult.hpp"
#include "DecomposeOption.hpp"
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
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(DenseTensor<2, DATATYPE, mtype, device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec, DecomposeOption& option){
    if(option.algorithm_type == DecomposeMethod::Direct){
        return evd(tensor, eigvec);
    }
    else if(option.algorithm_type == DecomposeMethod::Davidson){
        DenseTensorOperations<mtype, device>* basic_op = new DenseTensorOperations<mtype,device>(tensor);
        auto return_val = davidson(basic_op, eigvec, option);
        delete basic_op;
        return return_val;
    }
    else{
        std::cout << int(option.algorithm_type) << " should be one of DecomposMethod type. The given algorithm is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(SparseTensor<2, DATATYPE, mtype, device>& tensor, DenseTensor<2, DATATYPE, mtype, device>* eigvec, DecomposeOption& option){
    if(option.algorithm_type == DecomposeMethod::Davidson){
        SparseTensorOperations<mtype, device>* basic_op = new SparseTensorOperations(tensor);
        auto return_val = davidson(basic_op, eigvec, option);
        delete basic_op;
        return return_val;
    }
    else{
        std::cout << int(option.algorithm_type) << " should be one of DecomposMethod type. The given algorithm is not implemented" << std::endl;
        exit(1);
    }
};


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
std::unique_ptr<DecomposeResult<DATATYPE> > decompose(TensorOperations<mtype,device>* operations, DenseTensor<2, DATATYPE, mtype, device>* eigvec, DecomposeOption& option){
    if(option.algorithm_type == DecomposeMethod::Davidson){
        return davidson(operations, eigvec, option);
    }
    else{
        std::cout << int(option.algorithm_type) << " should be one of DecomposMethod type. The given algorithm is not implemented" << std::endl;
        exit(1);
    }
};


}
