#pragma once
#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{

// cublasHandle_t handle & cudaStream_t stream are defined in Utility.hpp

template<>
std::unique_ptr<Comm<DEVICETYPE::CUDA> > create_comm(int argc, char *argv[]){

    return std::make_unique< Comm<DEVICETYPE::CUDA> >( 0, 1);
}

template<>
Comm<DEVICETYPE::CUDA>::Comm(int rank, int world_size): rank(rank), world_size(world_size)
{
    if (handle ==NULL){
        auto status1 =cublasCreate(&SE::handle);
        assert (CUBLAS_STATUS_SUCCESS==status1);
    }
    if (stream==NULL){ 
        auto status2 = cudaStreamCreate(&stream);
        assert ( cudaSuccess == status2);
    }
    count+=1;

    return;
}

template<>
Comm<DEVICETYPE::CUDA>::~Comm(){
    count-=1;
    if (count==0 && handle!=NULL){
        auto status1 = cublasDestroy(SE::handle);
        assert (CUBLAS_STATUS_SUCCESS==status1);
        handle=NULL;
    }
    if(count==0 && stream!=NULL){
        // Finalize CUDA stream
        auto status2 = cudaStreamDestroy(stream);
        assert ( cudaSuccess == status2);
        stream=NULL;
    }
}

template<>
template <typename DATATYPE>
void Comm<DEVICETYPE::CUDA>::allreduce(const DATATYPE *src, int count, DATATYPE *trg, OPTYPE op) const{
    memcpy<DATATYPE, DEVICETYPE::CUDA> (trg,src,count,COPYTYPE::DEVICE2DEVICE);
    return;
}

template<>
template <typename DATATYPE>
void Comm<DEVICETYPE::CUDA>::alltoall(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::CUDA> (trg,src,sendcount,COPYTYPE::DEVICE2DEVICE);
    return ;
}

template<>
template <typename DATATYPE>
void Comm<DEVICETYPE::CUDA>::allgather(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::CUDA> (trg,src,sendcount,COPYTYPE::DEVICE2DEVICE);
  
}

}
