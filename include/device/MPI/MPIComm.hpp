#pragma once
#include <mpi.h>
#include "../../Comm.hpp"

namespace SE{

MPI_Comm mpi_comm = MPI_COMM_WORLD;
//template<>
//class Comm<computEnv::MPI>{
//    void initialize(int argc, char *argv[]);
//public:
//    MPI_Comm mpi_comm;
//    size_t rank = 0;           // Process rank
//    size_t world_size = 1;     // Total number of processes in the job
//    //Comm(MPI_Comm new_communicator);
//    //Comm(int argc, char *argv[]);
//    Comm() {};
//    ~Comm();
//
//    void initialize(int argc, char *argv[]);
//    void barrier();
//    
//    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SE_op op);
//    template <typename datatype> void alltoall (datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
//    template <typename datatype> void allgather(datatype* src, size_t sendcount, datatype* trg, size_t recvcount);
//};
//
/*
template<>
Comm<MPI>::Comm(MPI_Comm new_communicator) : mpi_comm(new_communicator){
    // Initialize MPI (assuming MPI_Init has been called before this)
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}
*/

template<>
void Comm<computEnv::MPI>::initialize(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}

template<>
Comm<computEnv::MPI>::~Comm(){
    if(MPI_Finalize() == MPI_SUCCESS){
        //std::cout << "The MPI routine MPI_Finalize succeeded." << std::endl;
    }
    else{
        std::cout << "The MPI routine MPI_Finalize failed." << std::endl;
    }
}

template<>
void Comm<computEnv::MPI>::barrier(){
    MPI_Barrier(MPI_COMM_WORLD);
}

template <> template<>
void Comm<computEnv::MPI>::allreduce<double>(const double *src, size_t count, double *trg, SE_op op){
    switch (op){
        case SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD); break;
        case PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD); break;
        case MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD); break;
        case MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}
template <> template<>
void Comm<computEnv::MPI>::alltoall<double>(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

template <> template<>
void Comm<computEnv::MPI>::allgather<double>(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, MPI_COMM_WORLD);
}

}
