#pragma once
#include <iostream>
#include <string>
#include "Device.hpp"
//#include <mpi.h>

namespace TH{
enum TH_op{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
};

template<typename device>
class Comm{
private:
    size_t rank = 0;           // Process rank
    size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 1;     // Total number of processes in the job
    std::string comm_protocol; // Communication protocol (e.g., "mpi," "nccl," "gloo," etc.)

public:
    Comm(){};
//Comm(std::string protocol): comm_protocol(std::move(protocol)){};
    //Comm(MPI_Comm new_communicator);
    //Comm(int argc, char *argv[]);
    ~Comm(){};

    const size_t get_rank(){ return rank; };
    const size_t get_local_rank(){ return local_rank; };
    const size_t get_world_size(){ return world_size; };
    const std::string get_comm_protocol(){ return comm_protocol; };

    void barrier();
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);

};
}
