#pragma once
#include <cassert>
#include <iostream>
#include "../../Comm.hpp"
#include "Utility.hpp"

namespace SE{
class MKLCommInp: public CommInp<DEVICETYPE::MKL> 
{
	public:
    	std::unique_ptr<Comm<DEVICETYPE::MKL> > create_comm(){
			return std::make_unique< Comm<DEVICETYPE::MKL> >( 0, 1 );
        }
};

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allreduce(const DATATYPE *src, int count, DATATYPE *trg, OPTYPE op) const{
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, count);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::alltoall(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allgather(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::allgatherv(DATATYPE *src, int sendcount, DATATYPE *trg, int* recvcount) const{
    assert(sendcount == recvcount[0]);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::scatterv(DATATYPE *src, int* sendcounts, DATATYPE *trg, int recvcount, int root) const{
    assert(sendcounts[0] == recvcount);
    assert(root == 0);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, recvcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::MKL>::alltoallv(DATATYPE *src, int* sendcounts, DATATYPE *trg, int* recvcounts) const{
    assert(sendcounts[0] == recvcounts[0]);
    memcpy<DATATYPE, DEVICETYPE::MKL>(trg, src, recvcounts[0]);
}

template<>
std::unique_ptr<CommInp<DEVICETYPE::MKL> > Comm<DEVICETYPE::MKL>::generate_comm_inp() const{
	return std::make_unique<MKLCommInp >();
}


}
