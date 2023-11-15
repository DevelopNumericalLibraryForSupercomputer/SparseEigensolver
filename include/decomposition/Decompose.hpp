#pragma once
#include <memory>
#include <functional>
#include "DirectSolver.hpp"
#include "IterativeSolver.hpp"

namespace SE{

/*
template<typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult<datatype, 2, comm, map> > decompose(std::function<DenseTensor<datatype,2,comm,map> (DenseTensor<datatype,2,comm,map>) >& matvec, std::string method)
{static_assert(false, "not implemented yet"); };
*/
//std::unique_ptr<DecomposeResult<datatype, 2, comm, map> > decompose(DenseTensor<datatype,2,comm,map>& tensor, std::string method)
template<typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult<datatype> > decompose(DenseTensor<datatype,2,comm,map>& tensor, std::string method)
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
}


template<typename datatype, typename comm, typename map> 
std::unique_ptr<DecomposeResult<datatype> > decompose(SparseTensor<datatype,2,comm,map>& tensor, std::string method)
{
    if(method == "davidson"){
        return davidson(tensor);
    }
    else{
        std::cout << method << " is not implemented" << std::endl;
        exit(1);
    }
}
    


}
