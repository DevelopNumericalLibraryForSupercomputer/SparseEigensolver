#pragma once
#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <typeinfo>

#include "Type.hpp"

namespace SE{

template<DEVICETYPE device>
class CommInp;

template<DEVICETYPE device>
class Comm{
public:
    Comm(int rank, int world_size, std::array<int,2> nprow={1,1}): rank(rank), world_size(world_size), nprow(nprow) { assert(nprow[1]*nprow[0]==world_size); count_comm++;};
    Comm(){count_comm++;};
    ~Comm(){count_comm--;};
    void finalize(){};
    //const int get_rank(){ return rank; };
    //const int get_world_size(){ return world_size; };

    void barrier() const{};
    std::unique_ptr<Comm<device>> clone ()const {return std::make_unique< Comm<device> > (rank, world_size, nprow);};
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
	std::array<int,2> get_nprow()const {return nprow;};
	std::unique_ptr<CommInp<device> > generate_comm_inp() const;
	static int get_count_comm(){return count_comm;};
private:
    int rank = 0;           // Process rank
    //int local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    int world_size = 1;     // Total number of processes in the job
	std::array<int,2> nprow={0,0};
	
protected:
	static int count_comm;
    //inline static int count_comm =0;
};
template<DEVICETYPE device>
int Comm<device>::count_comm=0;
// helper function 
template<DEVICETYPE device>
std::ostream &operator<<(std::ostream &os, Comm<device> const &comm) { 
    return os << "Comm<" << typeid(device).name() << ">"<<std::endl ;
}


template<DEVICETYPE device>
class CommInp
{
	public:
		virtual std::unique_ptr<Comm<device> > create_comm()=0;
        virtual ~CommInp(){};
}; 

//{
//    std::cout << "empty comm" << std::endl;
//    return std::make_unique< Comm<device> >();
//};



}
