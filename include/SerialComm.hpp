#pragma once
#include "Comm.hpp"

namespace TH{
class SerialComm: public Comm{
public:
    SerialComm(){};
    SerialComm(const std::string &protocol);
    SerialComm(int argc, char *argv[], const std::string &protocol) : SerialComm(protocol){};
    ~SerialComm();

    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){std::cout << "not implemented" << std::endl;};
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount){std::cout << "not implemented" << std::endl;};
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount){std::cout << "not implemented" << std::endl;};
};

}