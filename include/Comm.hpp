// communicator
// 참고: https://sooftware.io/big-model2/
// Chat-GPT 3.5
// To-do : define operator. op link to MPI
//         connect MPI datatype with datatype variable.
#pragma once
#include <iostream>
#include <string>
#include <mpi.h>
//#include <nccl.h>

namespace TensorHetero{
enum TH_op{ //operator
    TH_max,
    TH_min,
    TH_sum,
    TH_prod
};

class Comm{
private:
    // How can I initialize const size_t type variable using MPI_Comm_rank function????
    size_t rank = 0;           // Process rank
    size_t local_rank = 0;     // Local rank within a node (e.g., GPU ID)
    size_t world_size = 0;     // Total number of processes in the job
    std::string comm_protocol; // Communication protocol (e.g., "mpi," "nccl," "gloo," etc.)
    /*
    ncclUniqueId nccl_id;
    ncclComm_t nccl_communicator;
    */
public:
    Comm(){};
    Comm(const std::string &protocol);
    Comm(int argc, char *argv[], const std::string &protocol);
    ~Comm();

    const size_t get_rank(){ return rank; };
    const size_t get_local_rank(){ return local_rank; };
    const size_t get_world_size(){ return world_size; };
    const std::string get_comm_protocol(){ return comm_protocol; };

    template <typename datatype> void allreduce(datatype *src, size_t count, datatype *trg, enum TH_op op);
    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
};

Comm::Comm(const std::string &protocol) : comm_protocol(protocol){
    if (protocol == "mpi"){
        // Initialize MPI (assuming MPI_Init has been called before this)
        int tmp_rank, tmp_world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
        rank = tmp_rank;
        world_size = tmp_world_size;
    }
    else if (protocol == "nccl") {
        /* Check the following lines are correct
        // Initialize NCCL (assuming NCCL has been initialized before this)
        ncclComm_t nccl_comm;
        ncclGetUniqueId(&nccl_id);
        ncclCommInitRank(&nccl_comm, world_size, nccl_id, rank);
        // Assign the local rank
        ncclCommUserRank(nccl_comm, &local_rank);
        // Assign the communicator and initialize NCCL
        nccl_communicator = nccl_comm;
        */
        std::cout << "nccl is not implemented" << std::endl;
    }
}

Comm::Comm(int argc, char *argv[], const std::string &protocol) : comm_protocol(protocol){
    if (protocol == "mpi"){
        // Initialize MPI (assuming MPI_Init has been called before this)
        MPI_Init(&argc, &argv);
        int tmp_rank, tmp_world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
        rank = tmp_rank;
        world_size = tmp_world_size;
        std::cout << "rank = " << tmp_rank << ", world_size = " << tmp_world_size << std::endl;
    }
    else {
        std::cout << "nccl is not implemented" << std::endl;
    }
}

Comm::~Comm(){
    if (comm_protocol == "mpi"){
        // Finialize MPI
        MPI_Finalize();
    }
    if (comm_protocol == "nccl"){
        /*
        // Finalize NCCL
        ncclCommDestroy(nccl_communicator);
        */
    }
}


template <typename datatype>
void Comm::allreduce(datatype *src, size_t count, datatype *trg, TH_op op){
    std::cout << "not implemented" << std::endl;
}

template <>
void Comm::allreduce(double *src, size_t count, double *trg, TH_op op){
    if(comm_protocol == "mpi"){
        switch (op){
            case TH_sum:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD); break;
            case TH_prod: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD); break;
            case TH_max:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD); break;
            case TH_min:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD); break;
            default: std::cout << "WRONG OPERATION TYPE" << std::endl;
        }
    }
}

template <typename datatype>
void Comm::alltoall(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void Comm::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
//int MPI_Alltoall (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    if (comm_protocol == "mpi"){
        MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
    }
}

template <typename datatype>
void Comm::allgather(datatype *src, size_t sendcount, datatype *trg, size_t recvcount){
    std::cout << "not implemented" << std::endl;
}
template <>
void Comm::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
//int MPI_Alltoall (void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    if (comm_protocol == "mpi"){
        MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
    }
}
}