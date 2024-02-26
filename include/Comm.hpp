#pragma once
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <typeinfo>

#include "Type.hpp"

namespace SE{

template<DEVICETYPE device>
class Comm{
public:
    Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size) {};
    Comm(){};
    ~Comm(){};
    void finalize(){};
    //const size_t get_rank(){ return rank; };
    //const size_t get_world_size(){ return world_size; };

    void barrier() const{};
    Comm<device>* clone ()const {return new Comm(rank, world_size);};
    //template <typename DATATYPE> void send(DATATYPE* src, size_t sendcount, size_t recv_rank);
    //template <typename DATATYPE> void recv(DATATYPE* src, size_t sendcount, size_t recv_rank);
    

    //template <typename DATATYPE> void reduce(const DATATYPE *src, size_t count, DATATYPE *trg, SEop op, int root);
    //template <typename DATATYPE> void gather(DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t recvcount);

    template <typename DATATYPE> void allreduce(const DATATYPE *src, size_t count, DATATYPE *trg, OPTYPE op) const;
    template <typename DATATYPE> void alltoall (DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t recvcount) const;
    template <typename DATATYPE> void allgather(DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t recvcount) const;

    template <typename DATATYPE> void allgatherv(DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t* recvcounts) const;
    template <typename DATATYPE> void scatterv(DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t recvcount, size_t root) const;
    template <typename DATATYPE> void alltoallv (DATATYPE* src, size_t* sendcounts, DATATYPE* trg, size_t* recvcounts) const;

    size_t get_rank() const {return rank;};
    size_t get_world_size() const {return world_size;};
    private:
    size_t rank = 0;           // Process rank
    //size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 1;     // Total number of processes in the job
protected:
    inline static size_t count =0;
};

// helper function 
template<DEVICETYPE device>
std::ostream &operator<<(std::ostream &os, Comm<device> const &comm) { 
    return os << "Comm<" << typeid(device).name() << ">"<<std::endl ;
}

template<DEVICETYPE device>
std::unique_ptr<Comm<device> > create_comm(int argc, char *argv[]);
//{
//    std::cout << "empty comm" << std::endl;
//    return std::make_unique< Comm<device> >();
//};



}
