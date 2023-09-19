#pragma once
#include <mpi.h>
#include "Comm.hpp"

namespace TH{
class MPIComm: public Comm{
public:
    MPIComm(){};
    MPIComm(const std::string &protocol);
    MPIComm(int argc, char *argv[], const std::string &protocol);
    ~MPIComm();

    void barrier();
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

MPIComm::MPIComm(const std::string &protocol){
    // Initialize MPI (assuming MPI_Init has been called before this)
    assert( protocol.compare("CPU") || protocol.compare("cpu"));
    comm_protocol = protocol;
    this->device = CPU();
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

MPIComm::MPIComm(int argc, char *argv[], const std::string &protocol){
    assert( protocol.compare("CPU") || protocol.compare("cpu"));
    comm_protocol = protocol;
    MPI_Init(&argc, &argv);
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

MPIComm::~MPIComm(){
    MPI_Finalize();
}

void MPIComm::barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename datatype>
void MPIComm::allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <>
void MPIComm::allreduce(const double *src, size_t count, double *trg, enum TH_op op){
    assert(get_device_info() == "CPU");
    switch (op){
        case SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD); break;
        case PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD); break;
        case MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD); break;
        case MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}

template <typename datatype>
void MPIComm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}
template <>
void MPIComm::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(get_device_info() == "CPU");
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

template <typename datatype>
void MPIComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}
template <>
void MPIComm::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    assert(get_device_info() == "CPU");
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

}