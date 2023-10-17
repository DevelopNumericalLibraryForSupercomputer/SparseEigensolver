#pragma once
#include <iostream>
#include <string>
#include "Device.hpp"

namespace SE{
typedef enum{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
} SE_op;

template<PROTOCOL protocol = PROTOCOL::SERIAL>
class Comm{
    public:
        static const PROTOCOL _protocol=protocol;
        size_t rank = 0;           // Process rank
        //size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
        size_t world_size = 1;     // Total number of processes in the job
    
        //Comm(std::string protocol): comm_protocol(std::move(protocol)){};
        //Comm(MPI_Comm new_communicator);
        //Comm(int argc, char *argv[]) {};
        Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size), protocol(protocol)  {};
        void initialize() {};
        void initialize(int argc, char *argv[]) {};
    
        Comm(){ initialize(); };
        ~Comm(){};
    
        //const size_t get_rank(){ return rank; };
        //const size_t get_world_size(){ return world_size; };
    
        void barrier() {};
        template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SE_op op);
        template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
        template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

template <PROTOCOL protocol>
template <typename datatype>
inline void Comm<protocol>::allreduce(const datatype *src, size_t count, datatype *trg, SE_op op){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <PROTOCOL protocol>
template <typename datatype>
inline void Comm<protocol>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <PROTOCOL protocol>
template <typename datatype>
inline void Comm<protocol>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

}
