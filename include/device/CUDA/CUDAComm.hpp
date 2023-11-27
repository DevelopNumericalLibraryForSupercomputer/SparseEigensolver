#pragma once
#include <iostream>
#include <string>
#include <stdexcept>
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{

cudaStream_t stream;

template<>
std::unique_ptr<Comm<CUDA> > createComm<CUDA >(int argc, char *argv[]){

    return std::make_unique< Comm<CUDA> >( 0, 1);
}

template<>
Comm<CUDA>::Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size)
{
    auto status =cublasCreate(&SE::cublasHandle);
    assert (CUBLAS_STATUS_SUCCESS==status);

    cudaStreamCreate(&stream);
    return;
}

template<>
Comm<CUDA>::~Comm(){
    //Finalize cublas 
    auto status = cublasDestroy(SE::cublasHandle);
    assert (CUBLAS_STATUS_SUCCESS==status);
    // Finalize CUDA stream
    cudaStreamDestroy(stream);
}

template<>
template <typename datatype>
void Comm<CUDA>::allreduce(const datatype *src, size_t count, datatype *trg, SEop op){
    memcpy<datatype, CUDA> (trg,src,count);
    return;
}

template<>
template <typename datatype>
void Comm<CUDA>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<datatype, CUDA> (trg,src,sendcount);
    return ;
}

template<>
template <typename datatype>
void Comm<CUDA>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<datatype, CUDA> (trg,src,sendcount);
  
}

}
