#pragma once
#include "../../Comm.hpp"
#include <cassert>
#include "Utility.hpp"

namespace SE{
template<Serial>
class Comm{
    size_t rank = 0;
    size_t world_size = 1;

    Comm(){};
    Comm(int argc, char *argv[]);
    Comm(size_t rank, size_t world_size);
    ~Comm();

    void barrier() {};

    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SE_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

template<>
Comm<Serial>::Comm(size_t rank, size_t world_size){
    assert(world_size == 1 && rank == 0);
}

template <>
void Comm<Serial>::allreduce(const double *src, size_t count, double *trg, SE_op op){
    memcpy<double, Serial>(trg, src, count);
}

template <>
void Comm<Serial>::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, Serial>(trg, src, sendcount);
}

template <>
void Comm<Serial>::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, Serial>(trg, src, sendcount);
}

}