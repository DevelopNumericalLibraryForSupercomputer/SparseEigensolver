#pragma once
#include <mpi.h>
#include "../../Comm.hpp"

namespace SE{

class MPIComm: public Comm{
public:
    //MPIComm(MPI_Comm new_communicator);
    //Comm(int argc, char *argv[]);
    ~MPIComm();
    const std::string comm_protocol = "MPI"; // Communication protocol (e.g., "mpi," "nccl," "gloo," etc.)
    //const MPI_Comm mpi_comm;
    void barrier();
    
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum SE_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};
/*
MPIComm::MPIComm(MPI_Comm new_communicator) : mpi_comm(new_communicator){
    // Initialize MPI (assuming MPI_Init has been called before this)
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

MPIComm::MPIComm(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}
*/

MPIComm::~MPIComm(){
    MPI_Finalize();
}

void MPIComm::barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename datatype>
void MPIComm::allreduce(const datatype *src, size_t count, datatype *trg, enum SE_op op){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <>
void MPIComm::allreduce(const double *src, size_t count, double *trg, enum SE_op op){
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
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

template <typename datatype>
void MPIComm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}
template <>
void MPIComm::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

}