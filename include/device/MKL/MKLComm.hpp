#pragma once
#include <cassert>

#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{
//template<>
/*
class Comm<computEnv::MKL>{
public:
    size_t rank = 0;
    size_t world_size = 1;
    Comm() {};
    ~Comm() {};

    void barrier() {};

    template <typename double> void allreduce(const double *src, size_t count, double *trg, SE_op op);
    template <typename double> void alltoall (double* src, size_t sendcount, double* trg, size_t recvcount);
    template <typename double> void allgather(double* src, size_t sendcount, double* trg, size_t recvcount);
};
*/
template<>
std::unique_ptr<Comm<MKL> > createComm<MKL>(int argc, char *argv[]){
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

}
