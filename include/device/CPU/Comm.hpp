#pragma once
#include <mpi.h>
#include "../../Comm.hpp"

namespace TH{
template <>
class Comm<TH::CPU>{
public:
    Comm(MPI_Comm new_communicator);
    Comm(int argc, char *argv[]);
    ~Comm();

    void barrier();
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);

    const size_t get_rank(){ return rank; };
    const size_t get_local_rank(){ return local_rank; };
    const size_t get_world_size(){ return world_size; };
    const std::string get_comm_protocol(){ return comm_protocol; };

private:
    MPI_Comm mpi_comm;
    size_t rank = 0;           // Process rank
    size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 1;     // Total number of processes in the job
    std::string comm_protocol = "MPI"; // Communication protocol (e.g., "mpi," "nccl," "gloo," etc.)
};

Comm<TH::CPU>::Comm(MPI_Comm new_communicator){
    // Initialize MPI (assuming MPI_Init has been called before this)
    comm_protocol = "MPI";
    mpi_comm = new_communicator;
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

Comm<TH::CPU>::Comm(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    mpi_comm = MPI_COMM_WORLD;
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

Comm<TH::CPU>::~Comm(){
    MPI_Finalize();
}

void Comm<TH::CPU>::barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

template <typename datatype>
void Comm<TH::CPU>::allreduce(const datatype *src, size_t count, datatype *trg, enum TH_op op){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}

template <>
void Comm<TH::CPU>::allreduce(const double *src, size_t count, double *trg, enum TH_op op){
    switch (op){
        case SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD); break;
        case PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD); break;
        case MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD); break;
        case MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}

template <typename datatype>
void Comm<TH::CPU>::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}
template <>
void Comm<TH::CPU>::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

template <typename datatype>
void Comm<TH::CPU>::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
    exit(-1);
}
template <>
void Comm<TH::CPU>::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

}