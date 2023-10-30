#pragma once
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include "Device.hpp"

namespace SE{
typedef enum{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
} SE_op;



template<computEnv comput_env = computEnv::BASE>
class Comm{
    public:
        static constexpr computEnv env = comput_env;

        Comm(size_t rank=0, size_t world_size=1): rank(rank), world_size(world_size) {};
        Comm(){ };
        ~Comm(){};
    
        //const size_t get_rank(){ return rank; };
        //const size_t get_world_size(){ return world_size; };
    
        void barrier() {};
        template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SE_op op);
        template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
        template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);

        size_t rank = 0;           // Process rank
        //size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
        size_t world_size = 1;     // Total number of processes in the job
    
};

// helper function 
template<computEnv comput_env>
std::ostream &operator<<(std::ostream &os, Comm<comput_env> const &comm) { 
    return os << "Comm<" << comm.env  << ">"<<std::endl ;
}
template<computEnv comput_env>
std::unique_ptr<Comm<comput_env> > createComm(int argc, char *argv[]);
}

