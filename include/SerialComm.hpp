#pragma once
#include "Comm.hpp"
#include <cassert>
#include "Device.hpp"
#include "Utility_include.hpp"

namespace TH{

class SerialComm: public Comm{
public:
    SerialComm();
    /*
    SerialComm(int argc, char *argv[], const std::string &protocol) : SerialComm(protocol){};
    */
    ~SerialComm();

    void barrier(){};
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

SerialComm::SerialComm(){
    this->comm_protocol = "CPU";
    this->device = CPU();
}

template <typename datatype>
void SerialComm::allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::allreduce(const double *src, size_t count, double *trg, enum TH_op op){
    assert(get_device_info() == "CPU");
    memcpy<double, CPU>(trg, src, count);
}

template <typename datatype>
void SerialComm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(get_device_info() == "CPU");
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

template <typename datatype>
void SerialComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void SerialComm::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(get_device_info() == "CPU");
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

}