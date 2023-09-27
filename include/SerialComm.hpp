#pragma once
#include "Comm.hpp"
#include <cassert>
#include "Device.hpp"
#include "Utility_include.hpp"

namespace SE{
class SerialComm: public Comm{
public: 
    const std::string comm_protocol = "Serial";
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum SE_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

template <typename datatype>
void SerialComm::allreduce(const datatype *src, size_t count, datatype *trg, enum SE_op op){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::allreduce(const double *src, size_t count, double *trg, enum SE_op op){
    memcpy<double, CPU>(trg, src, count);
}

template <typename datatype>
void SerialComm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

template <typename datatype>
void SerialComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

}