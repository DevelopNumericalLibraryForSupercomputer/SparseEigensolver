#pragma once
#include <mpi.h>
#include <cassert>
#include "../../Comm.hpp"
#include "Utility.hpp"
namespace SE{

MPI_Comm mpi_comm = MPI_COMM_WORLD;
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
//    template <typename DATATYPE> void allreduce(const DATATYPE *src, size_t count, DATATYPE *trg, SE_op op);
//    template <typename DATATYPE> void alltoall (DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t recvcount);
//    template <typename DATATYPE> void allgather(DATATYPE* src, size_t sendcount, DATATYPE* trg, size_t recvcount);
//};
//
/*
template<>
Comm<SEMpi>::Comm(MPI_Comm new_communicator) : mpi_comm(new_communicator){
    // Initialize MPI (assuming MPI_Init has been called before this)
    int tmp_rank, tmp_world_size;
    MPI_Comm_rank(mpi_comm, &tmp_rank);
    MPI_Comm_size(mpi_comm, &tmp_world_size);
    rank = tmp_rank;
    world_size = tmp_world_size;
}
*/

template<>
std::unique_ptr<Comm<SEMpi> > createComm(int argc, char *argv[]){
    std::cout << "MPI Comm" << std::endl;
    MPI_Init(&argc, &argv);
    int myRank ,nRanks;
    MPI_Comm_rank(mpi_comm, &myRank);
    MPI_Comm_size(mpi_comm, &nRanks);
    assert(nRanks>0);
    assert(myRank>=0);
    return std::make_unique< Comm<SEMpi> >( (size_t) myRank, (size_t) nRanks );
}

template<>
Comm<SEMpi>::~Comm(){
    if(MPI_Finalize() == MPI_SUCCESS){
        //std::cout << "The MPI routine MPI_Finalize succeeded." << std::endl;
    }
    else{
        std::cout << "The MPI routine MPI_Finalize failed." << std::endl;
    }
}

template<>
void Comm<SEMpi>::barrier() const{
    MPI_Barrier(mpi_comm);
}

/*
template <>
template <>
void Comm<SEMpi>::reduce(const size_t *src, size_t count, size_t *trg, SEop op, int root)
{
    switch (op){
        case SEop::SUM:  MPI_Reduce(src, trg, count, MPI_UNSIGNED_LONG_LONG, MPI_SUM,  root, mpi_comm); break;
        case SEop::PROD: MPI_Reduce(src, trg, count, MPI_UNSIGNED_LONG_LONG, MPI_PROD, root, mpi_comm); break;
        case SEop::MAX:  MPI_Reduce(src, trg, count, MPI_UNSIGNED_LONG_LONG, MPI_MAX,  root, mpi_comm); break;
        case SEop::MIN:  MPI_Reduce(src, trg, count, MPI_UNSIGNED_LONG_LONG, MPI_MIN,  root, mpi_comm); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }    
}
*/

template <> 
template <> 
void Comm<SEMpi>::allreduce(const double *src, size_t count, double *trg, SEop op) const{
    switch (op){
        case SEop::SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  mpi_comm); break;
        case SEop::PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, mpi_comm); break;
        case SEop::MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  mpi_comm); break;
        case SEop::MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  mpi_comm); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}
template <> 
template <> 
void Comm<SEMpi>::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount) const{
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<SEMpi>::alltoallv(double *src, size_t* sendcounts, double *trg, size_t* recvcounts) const{
    // displs are automatically generated
    int sdispls[world_size];
    int int_sendcounts[world_size];
    int rdispls[world_size];
    int int_recvcounts[world_size*world_size];

    sdispls[0] = 0;
    rdispls[0] = 0;
    int_sendcounts[0] = (int)sendcounts[0];
    int_recvcounts[0] = (int)recvcounts[0];
    for(int i=1;i<world_size;i++){
        sdispls[i] = sdispls[i-1] + (int)sendcounts[i-1];
        int_sendcounts[i] = (int)sendcounts[i];
        rdispls[i] = rdispls[i-1] + (int)recvcounts[i-1];
        int_recvcounts[i] = (int)recvcounts[i];
    }
    MPI_Alltoallv(src, int_sendcounts, sdispls, MPI_DOUBLE, trg, int_recvcounts, rdispls, MPI_DOUBLE, mpi_comm );
}

template <> 
template <> 
void Comm<SEMpi>::allgather(double *src, size_t sendcount, double *trg, size_t recvcount) const{
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<SEMpi>::allgatherv(double *src, size_t sendcount, double *trg, size_t* recvcounts) const{
    // displs are automatically generated using recvcount
    int displs[world_size];
    int int_recvcounts[world_size];
    displs[0] = 0;
    int_recvcounts[0] = (int)recvcounts[0];
    for(int i=1;i<world_size;i++){
        displs[i] = displs[i-1] + (int)recvcounts[i-1];
        int_recvcounts[i] = (int)recvcounts[i];
    }
    MPI_Allgatherv(src, (int)sendcount, MPI_DOUBLE, trg, int_recvcounts, displs, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<SEMpi>::scatterv(double *src, size_t* sendcounts, double *trg, size_t recvcount, size_t root) const{
    // displs are automatically generated using recvcount
    int displs[world_size];
    int int_sendcounts[world_size];
    displs[0] = 0;
    int_sendcounts[0] = (int)sendcounts[0];
    for(int i=1;i<world_size;i++){
        displs[i] = displs[i-1] + (int)sendcounts[i-1];
        int_sendcounts[i] = (int)sendcounts[i];
    }
    MPI_Scatterv(src, int_sendcounts, displs, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, root, mpi_comm);
}




}
