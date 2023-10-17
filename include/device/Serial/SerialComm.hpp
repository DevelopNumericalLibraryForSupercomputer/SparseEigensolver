#pragma once
#include "../../Comm.hpp"
#include <cassert>
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
template<> template<>
void Comm<computEnv::MKL>::allreduce<double>(const double *src, size_t count, double *trg, SE_op op){
    memcpy<double, computEnv::MKL>(trg, src, count);
}

template<> template<>
void Comm<computEnv::MKL>::alltoall<double>(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, computEnv::MKL>(trg, src, sendcount);
}

template<> template<>
void Comm<computEnv::MKL>::allgather<double>(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, computEnv::MKL>(trg, src, sendcount);
}

}
