// communicator
#pragma once
#include <string>

namespace TensorHetero{
template<typename datatype, typename device>
class Comm{
public:
    const size_t rank = 0;
    const size_t local_rank = 0;
    const size_t world_size = 0;

    const std::string comm_protocol; // mpi, nccl, ...

    //Comm(){};

    //virtual void allreduce<typename datatype>(datatype* src, std::string Operator) = 0;
    //virtual void all2all<typename datatype>(datatype* src, datatype* trg) = 0;
    //virtual void allgather<typename datatype> (datatype* src, datatype* trg) = 0;
};
}