#pragma once
#include <cassert>
#include <iostream>
#include "../../Comm.hpp"
#include "Utility.hpp"
#include "device/sycl/LinearOp.hpp"

namespace SE{

sycl::device get_device(const int i){

    std::vector<sycl::device> devices;
    try {
        devices = sycl::device::get_devices();
        //for (const auto& device : devices) {
        //    std::cout << "Device: " << device.get_info<sycl::info::device::name>() << "\n";
        //}
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL exception caught: " << e.what() << "\n";
    }
    return devices[i];
}

class SYCLCommInp: public CommInp<DEVICETYPE::SYCL> 
{
	public:
        SYCLCommInp(const sycl::device device){
            set_device(device);
        }
    	std::unique_ptr<Comm<DEVICETYPE::SYCL> > create_comm(){
			return std::make_unique< Comm<DEVICETYPE::SYCL> >( 0, 1 );
        }
};

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::allreduce(const DATATYPE *src, int count, DATATYPE *trg, OPTYPE op) const{
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, count);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::alltoall(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::allgather(DATATYPE *src, int sendcount, DATATYPE *trg, int recvcount) const{
    assert(sendcount == recvcount);
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::allgatherv(DATATYPE *src, int sendcount, DATATYPE *trg, int* recvcount) const{
    assert(sendcount == recvcount[0]);
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, sendcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::scatterv(DATATYPE *src, int* sendcounts, DATATYPE *trg, int recvcount, int root) const{
    assert(sendcounts[0] == recvcount);
    assert(root == 0);
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, recvcount);
}

template<>
template<typename DATATYPE>
void Comm<DEVICETYPE::SYCL>::alltoallv(DATATYPE *src, int* sendcounts, DATATYPE *trg, int* recvcounts) const{
    assert(sendcounts[0] == recvcounts[0]);
    memcpy<DATATYPE, DEVICETYPE::SYCL>(trg, src, recvcounts[0]);
}

template<>
std::unique_ptr<CommInp<DEVICETYPE::SYCL> > Comm<DEVICETYPE::SYCL>::generate_comm_inp() const{
	return std::make_unique<SYCLCommInp >(p_que->get_device());
}


}
