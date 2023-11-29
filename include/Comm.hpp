#pragma once
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include "Device.hpp"
#include <typeinfo>

namespace SE{
enum class SEop{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
};

template<typename computEnv>
class Comm{
public:
    using SEenv = ComputEnv;
    Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size) {};
    Comm(){ };
    ~Comm(){};

    //const size_t get_rank(){ return rank; };
    //const size_t get_world_size(){ return world_size; };

    void barrier() const{};

    //template <typename datatype> void send(datatype* src, size_t sendcount, size_t recv_rank);
    //template <typename datatype> void recv(datatype* src, size_t sendcount, size_t recv_rank);
    

    //template <typename datatype> void reduce(const datatype *src, size_t count, datatype *trg, SEop op, int root);
    //template <typename datatype> void gather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);

    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SEop op) const;
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount) const;
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount) const;

    template <typename datatype> void allgatherv(datatype* src, size_t sendcount, datatype* trg, size_t* recvcounts) const;
    template <typename datatype> void scatterv(datatype* src, size_t* sendcounts, datatype* trg, size_t recvcount, size_t root) const;
    template <typename datatype> void alltoallv (datatype* src, size_t* sendcounts, datatype* trg, size_t* recvcounts) const;

    size_t get_rank() const {return rank;};
    size_t get_world_size() const {return world_size;};
    private:
    size_t rank = 0;           // Process rank
    //size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 1;     // Total number of processes in the job

};

// helper function 
template<typename computEnv>
std::ostream &operator<<(std::ostream &os, Comm<computEnv> const &comm) { 
    return os << "Comm<" << typeid(computEnv).name() << ">"<<std::endl ;
}

template<typename computEnv>
std::unique_ptr<Comm<computEnv> > createComm(int argc, char *argv[]){
    std::cout << "empty comm" << std::endl;
    return std::make_unique< Comm<computEnv> >();
};



}
