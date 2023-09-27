#pragma once
#include <iostream>
#include <string>
//#include "nccl.h"
#include "../../Comm.hpp"

namespace TH{
class NcclComm: public Comm{
public:
    NcclComm(){};
    NcclComm(const std::string &protocol);
    NcclComm(int argc, char *argv[], const std::string &protocol);
    ~NcclComm();

    template <typename datatype> void allreduce(datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

NcclComm::NcclComm(const std::string &protocol) : Comm(protocol){
    /* Check the following lines are correct
    // Initialize NCCL (assuming NCCL has been initialized before this)
    ncclComm_t nccl_comm;
    ncclGetUniqueId(&nccl_id);
    ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);
    // Assign the local rank
    ncclCommUserRank(nccl_comm, &local_rank);
    // Assign the communicator and initialize NCCL
    nccl_communicator = nccl_comm;
    */
    std::cout << "nccl is not implemented" << std::endl;
}

NcclComm::NcclComm(int argc, char *argv[], const std::string &protocol) : Comm(protocol){
    std::cout << "nccl is not implemented" << std::endl;
}

NcclComm::~NcclComm(){
    /*
    // Finalize NCCL
    ncclCommDestroy(nccl_communicator);
    */
}

template <typename datatype>
void NcclComm::allreduce(datatype *src, size_t count, datatype *trg, TH_op op){
    std::cout << "not implemented" << std::endl;
}

template <typename datatype>
void NcclComm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}

template <typename datatype>
void NcclComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
}