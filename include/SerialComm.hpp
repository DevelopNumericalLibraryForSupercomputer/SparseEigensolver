#pragma once
#include "Comm.hpp"

namespace TH{

template<typename device>
class SerialComm: public Comm{
public:
    SerialComm(){};
    SerialComm(const std::string &protocol);
    SerialComm(int argc, char *argv[], const std::string &protocol) : SerialComm(protocol){};
    ~SerialComm();

    void barrier(){};
    template <typename datatype, typename device> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype, typename device> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype, typename device> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};


template <typename datatype, typename device>
void SerialComm::allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){
    std::cout << "not implemented" << std::endl;
}
template <typename device>
void SerialComm::allreduce(const double *src, size_t count, double *trg, enum TH_op op){
    assert(device.get_device_info() == "CPU");
    memcpy<double, CPU>(trg, src, count);
}

template <typename datatype, typename device>
void SerialComm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <typename device>
void SerialComm::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(device.get_device_info() == "CPU");
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

template <typename datatype, typename device>
void SerialComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <typename datatype, typename device>
void SerialComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    assert(device.get_device_info() == "CPU");
    assert(sendcount == recvcount);
    memcpy<double, CPU>(trg, src, sendcount);
}

}