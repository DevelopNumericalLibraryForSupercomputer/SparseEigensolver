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

template<computEnv comput_env = computEnv::MKL>
class Comm{
    public:
        static const computEnv env = comput_env;
        size_t rank = 0;           // Process rank
        //size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
        size_t world_size = 1;     // Total number of processes in the job
    
        //Comm(std::string comput_env): comm_comput_env(std::move(comput_env)){};
        //Comm(MPI_Comm new_communicator);
        //Comm(int argc, char *argv[]) {};
        Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size) {};
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

template <computEnv comput_env>
template <typename datatype>
inline void Comm<comput_env>::allreduce(const datatype *src, size_t count, datatype *trg, SE_op op){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <computEnv comput_env>
template <typename datatype>
inline void Comm<comput_env>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <computEnv comput_env>
template <typename datatype>
inline void Comm<comput_env>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

}
