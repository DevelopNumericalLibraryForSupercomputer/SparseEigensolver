#pragma once
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{
template<>
std::unique_ptr<Comm<SEMkl> > createComm(int argc, char *argv[]){
    std::cout << "MKL Comm" << std::endl;
    return std::make_unique< Comm<SEMkl> >( 0, 1 );
}

template<>
template<typename datatype>
void Comm<SEMkl>::allreduce(const datatype *src, size_t count, datatype *trg, SEop op) const{
    memcpy<datatype, SEMkl>(trg, src, count);
}

template<>
template<typename datatype>
void Comm<SEMkl>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount) const{
    assert(sendcount == recvcount);
    memcpy<datatype, SEMkl>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<SEMkl>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount) const{
    assert(sendcount == recvcount);
    memcpy<datatype, SEMkl>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<SEMkl>::allgatherv(datatype *src, size_t sendcount, datatype *trg, size_t* recvcount) const{
    assert(sendcount == recvcount[0]);
    memcpy<datatype, SEMkl>(trg, src, sendcount);
}

template<>
template<typename datatype>
void Comm<SEMkl>::scatterv(datatype *src, size_t* sendcounts, datatype *trg, size_t recvcount, size_t root) const{
    assert(sendcounts[0] == recvcount);
    assert(root == 0);
    memcpy<datatype, SEMkl>(trg, src, recvcount);
}

template<>
template<typename datatype>
void Comm<SEMkl>::alltoallv(datatype *src, size_t* sendcounts, datatype *trg, size_t* recvcounts) const{
    assert(sendcounts[0] == recvcounts[0]);
    memcpy<datatype, SEMkl>(trg, src, recvcounts[0]);
}

}
