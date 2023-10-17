#pragma once
#include <mpi.h>
#include <nccl.h>
#include <iostream>
#include <string>
#include <stdexcept>

#include "../../Comm.hpp"

namespace SE{

ncclComm_t nccl_comm;
//MPI_Comm mpi_comm = MPI_COMM_WORLD;
cudaStream_t stream;

Comm<PROTOCOL::NCCL>::Comm(int argc, char *argv[]) {
    std::cout << "nccl is not implemented" << std::endl;
    int myRank, nRanks;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
    ncclUniqueId id;
    if (myRank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclCommInitRank(&nccl_comm, nRanks, id, myRank);
    cudaStreamCreate(&stream);

}

Comm<PROTOCOL::NCCL>::~Comm(){
    // Finalize NCCL
    ncclCommDestroy(nccl_comm);
    // Finalize MPI
    MPI_Finalize();
}

template<typename datatype>
ncclDataType_t check_type(){
    if ( std::is_same<datatype, double> ){
        return ncclDouble;
    }
    else if (std::is_same<datatype, float> ){
        return ncclFloat;
    }
    throw std::runtime_error("NCCL does not support the type ");
}
ncclRedOp_t check_op(SE_op op){
    switch (op){
        case SUM: return ncclSum;
        case PROD: return ncclProd;
        case MIN: return ncclMin;
        case MAX: return ncclMax;
        default: throw std::runtime_error("CUDAComm check_op") ;
    }
}
template <typename datatype>
void Comm<PROTOCOL::NCCL>::allreduce(datatype *src, size_t count, datatype *trg, SE_op op){

    ncclDataType_t __datatype = check_type<datatype>();
    ncclRedOp_t    __op       = check_op(op);
    auto result = ncclAllreduce( (void*) src, (void*) trg, count, __datatype, __op,  nccl_comm);
    cudaStreamSynchronize(stream);
    return;
}

template <typename datatype>
void Comm<PROTOCOL::NCCL>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    ncclDataType_t __datatype = check_type<datatype>();
    int nRanks;
    ncclCommCount(nccl_comm, &nRanks);
    size_t rankOffset = count * wordSize(type);
  
    ncclGroupStart();
    for (int r=0; r<nRanks; r++) {
        ncclSend( (void*) (sendbuff+r*rankOffset), count, __datatype, r, nccl_comm, stream);
        ncclRecv( (void*) (recvbuff+r*rankOffset), count, __datatype, r, nccl_comm, stream);
    }
    ncclGroupEnd();
    cudaStreamSynchronize(stream);
    return ;
}

template <typename datatype>
void Comm<PROTOCOL::NCCL>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    auto __datatype = check_type<datatype>();
    auto __op       = check_op(op);
    auto result     = ncclAllGather( (void*) sendbuff, (void*) recvbuff, sendcount, __datatype, nccl_comm, stream);
    cudaStreamSynchronize(stream);
    std::cout << "not implemented" << std::endl;
}

}
