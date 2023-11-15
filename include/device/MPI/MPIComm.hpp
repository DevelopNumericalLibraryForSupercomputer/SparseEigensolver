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
std::unique_ptr<Comm<MPI> > createComm<MPI>(int argc, char *argv[]){
    MPI_Init(&argc, &argv);
    int myRank ,nRanks;
    MPI_Comm_rank(mpi_comm, &myRank);
    MPI_Comm_size(mpi_comm, &nRanks);
    assert(nRanks>0);
    assert(myRank>=0);
    return std::make_unique< Comm<MPI> >( (size_t) myRank, (size_t) nRanks );
}

template<>
Comm<MPI>::~Comm(){
    if(MPI_Finalize() == MPI_SUCCESS){
        //std::cout << "The MPI routine MPI_Finalize succeeded." << std::endl;
    }
    else{
        std::cout << "The MPI routine MPI_Finalize failed." << std::endl;
    }
}

template<>
void Comm<MPI>::barrier(){
    MPI_Barrier(mpi_comm);
}

/*
template <>
template <>
void Comm<MPI>::reduce(const size_t *src, size_t count, size_t *trg, SEop op, int root)
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
void Comm<MPI>::allreduce(const double *src, size_t count, double *trg, SEop op){
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
void Comm<MPI>::alltoall(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<MPI>::allgather(double *src, size_t sendcount, double *trg, size_t recvcount){
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<MPI>::allgatherv(double *src, int sendcount, double *trg, int* recvcounts){ // displs are automatically generated using recvcount
    int displs[world_size];
    displs[0] = 0;
    for(int i=1;i<world_size;i++){
        displs[i] = displs[i-1] + (int)recvcounts[i-1];
    }
    /*
    std::cout << "src : rank " << rank;
    for(int i=0;i<sendcount;i++){
        std::cout << " " <<src[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "recvcounts : ";
    for(int i=0;i<world_size;i++){
        std::cout << recvcounts[i] << ' ';
    }
    std::cout << std::endl;
    
    std::cout << "displ : ";
    for(int i=0;i<world_size;i++){
        std::cout << displs[i] << ' ';
    }
    std::cout << std::endl;
    */
    MPI_Allgatherv(src, sendcount, MPI_DOUBLE, trg, recvcounts, displs, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<MPI>::scatterv(double *src, int* sendcounts, double *trg, int recvcount, size_t root){ // displs are automatically generated using recvcount
    int displs[world_size];
    displs[0] = 0;
    for(int i=1;i<world_size;i++){
        displs[i] = displs[i-1] + (int)sendcounts[i-1];
    }
    /*
    std::cout << "src : rank " << rank;
    for(int i=0;i<recvcount;i++){
        std::cout << " " <<src[i] << ' ';
    }
    std::cout << std::endl;
    std::cout << "sendcounts : ";
    for(int i=0;i<world_size;i++){
        std::cout << sendcounts[i] << ' ';
    }
    std::cout << std::endl;
    
    std::cout << "displ : ";
    for(int i=0;i<world_size;i++){
        std::cout << displs[i] << ' ';
    }
    std::cout << std::endl;
    */

    MPI_Scatterv(src, sendcounts, displs, MPI_DOUBLE, trg, recvcount, MPI_DOUBLE, root, mpi_comm);
}

}
