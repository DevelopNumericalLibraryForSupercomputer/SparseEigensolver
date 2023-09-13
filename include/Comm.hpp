#pragma once
#include <iostream>
#include <string>

namespace TensorHetero{
enum TH_op{ //operator
    TH_max,
    TH_min,
    TH_sum,
    TH_prod
};

class Comm{
protected:
    size_t rank = 0;           // Process rank
    size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 0;     // Total number of processes in the job
    std::string comm_protocol; // Communication protocol (e.g., "mpi," "nccl," "gloo," etc.)

public:
    Comm(){};
    Comm(const std::string &protocol):comm_protocol(protocol){};
    Comm(int argc, char *argv[], const std::string &protocol);
    ~Comm(){};

    const size_t get_rank(){ return rank; };
    const size_t get_local_rank(){ return local_rank; };
    const size_t get_world_size(){ return world_size; };
    const std::string get_comm_protocol(){ return comm_protocol; };

    virtual void barrier() = 0;
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);

};
}