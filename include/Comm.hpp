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
    Comm(int rank, int world_size): rank(rank), world_size(world_size) {};
    Comm(){};
    ~Comm(){};
    void finalize(){};
    //const int get_rank(){ return rank; };
    //const int get_world_size(){ return world_size; };

    void barrier() const{};
    Comm<device>* clone ()const {return new Comm(rank, world_size);};
    //template <typename DATATYPE> void send(DATATYPE* src, int sendcount, int recv_rank);
    //template <typename DATATYPE> void recv(DATATYPE* src, int sendcount, int recv_rank);
    

    //template <typename DATATYPE> void reduce(const DATATYPE *src, int count, DATATYPE *trg, SEop op, int root);
    //template <typename DATATYPE> void gather(DATATYPE* src, int sendcount, DATATYPE* trg, int recvcount);

    template <typename DATATYPE> void allreduce(const DATATYPE *src, int count, DATATYPE *trg, OPTYPE op) const;
    template <typename DATATYPE> void alltoall (DATATYPE* src, int sendcount, DATATYPE* trg, int recvcount) const;
    template <typename DATATYPE> void allgather(DATATYPE* src, int sendcount, DATATYPE* trg, int recvcount) const;

    template <typename DATATYPE> void allgatherv(DATATYPE* src, int sendcount, DATATYPE* trg, int* recvcounts) const;
    template <typename DATATYPE> void scatterv(DATATYPE* src, int* sendcounts, DATATYPE* trg, int recvcount, int root) const;
    template <typename DATATYPE> void alltoallv (DATATYPE* src, int* sendcounts, DATATYPE* trg, int* recvcounts) const;

    int get_rank() const {return rank;};
    int get_world_size() const {return world_size;};
    private:
    int rank = 0;           // Process rank
    //int local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    int world_size = 1;     // Total number of processes in the job
protected:
    inline static int count =0;
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
