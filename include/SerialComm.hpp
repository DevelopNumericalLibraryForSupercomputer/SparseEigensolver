#pragma once
#include "Comm.hpp"
#include <cassert>
#include "Device.hpp"
#include "Utility_include.hpp"

namespace TH{
template<>
class Comm<TH::Serial>{
public:
    Comm(){};
    /*
    SerialComm(int argc, char *argv[], const std::string &protocol) : SerialComm(protocol){};
    */
    ~Comm();

    void barrier(){};
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
private:
    size_t rank = 0;           // Process rank
    size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 1;     // Total number of processes in the job
    std::string comm_protocol = "Serial";
};

template <typename datatype>
void Comm<TH::Serial>::allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){
    std::cout << "not implemented" << std::endl;
}
template <>
void Comm<TH::Serial>::allreduce(const double *src, size_t count, double *trg, enum TH_op op){
    memcpy<double, CPU>(trg, src, count);
}

template <typename datatype>
void Comm<TH::Serial>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void Comm<TH::Serial>::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

template <typename datatype>
void Comm<TH::Serial>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void Comm<TH::Serial>::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

}