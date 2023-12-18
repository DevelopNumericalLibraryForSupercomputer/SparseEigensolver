#pragma once
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{
template<>
std::unique_ptr<Comm<DEVICETYPE::MKL> > create_comm(int argc, char *argv[]){
    std::cout << "MKL Comm" << std::endl;
    return std::make_unique< Comm<DEVICETYPE::MKL> >( 0, 1 );
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allreduce(const DATATYPE *src, size_t count, DATATYPE *trg, OPTYPE op) const{
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, count*sizeof(DATATYPE));
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::alltoall(DATATYPE *src, size_t sendcount, DATATYPE *trg, size_t recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount*sizeof(DATATYPE));
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allgather(DATATYPE *src, size_t sendcount, DATATYPE *trg, size_t recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount*sizeof(DATATYPE) );
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allgatherv(DATATYPE *src, size_t sendcount, DATATYPE *trg, size_t* recvcount) const{
    assert(sendcount == recvcount[0]);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount*sizeof(DATATYPE));
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::scatterv(DATATYPE *src, size_t* sendcounts, DATATYPE *trg, size_t recvcount, size_t root) const{
    assert(sendcounts[0] == recvcount);
    assert(root == 0);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, recvcount*sizeof(DATATYPE));
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::alltoallv(DATATYPE *src, size_t* sendcounts, DATATYPE *trg, size_t* recvcounts) const{
    assert(sendcounts[0] == recvcounts[0]);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, recvcounts[0]*sizeof(DATATYPE));
}

}
