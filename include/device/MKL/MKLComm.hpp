#pragma once
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{
template<>
std::unique_ptr<Comm<MKL> > createComm(int argc, char *argv[]){
    std::cout << "SERIALcomm" << std::endl;
    return std::make_unique< Comm<MKL> >( 0, 1 );
}

template<>
template<typename datatype>
void Comm<MKL>::allreduce(const datatype *src, size_t count, datatype *trg, SEop op){
    memcpy<datatype, MKL>(trg, src, count);
}

template<>
template<typename datatype>
void Comm<MKL>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<datatype, MKL>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<MKL>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<datatype, MKL>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<MKL>::allgatherv(datatype *src, size_t sendcount, datatype *trg, size_t* recvcount){
    assert(sendcount == recvcount[0]);
    memcpy<datatype, MKL>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<MKL>::scatterv(datatype *src, size_t* sendcounts, datatype *trg, size_t recvcount, size_t root){
    assert(sendcounts[0] == recvcount);
    assert(root == 0);
    memcpy<datatype, MKL>(trg, src, recvcount);
}

template<>
template<typename datatype>
void Comm<MKL>::alltoallv(datatype *src, size_t* sendcounts, datatype *trg, size_t* recvcounts){
    assert(sendcounts[0] == recvcounts[0]);
    memcpy<datatype, MKL>(trg, src, recvcounts[0]);
}

}
