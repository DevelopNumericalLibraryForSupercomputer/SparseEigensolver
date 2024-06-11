#pragma once
#include <mpi.h>
#include <cassert>
#include "../../Comm.hpp"
#include "Utility.hpp"

#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"

namespace SE{

MPI_Comm mpi_comm = MPI_COMM_NULL; // MPI_COMM_WORLD;
int count_comm = 0;

class MPICommInp: public CommInp<DEVICETYPE::MPI> 
{
	public:
    	std::unique_ptr<Comm<DEVICETYPE::MPI> > create_comm(){
        	const int i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
            int rank ,world_size;
        	int ictxt;
        
        	blacs_pinfo( &rank, &world_size );
            blacs_get( &i_negone, &i_zero, &ictxt );
            blacs_gridinit( &ictxt, "C", &this->nprow[0], &this->nprow[1] );
        
        	if (mpi_comm==MPI_COMM_NULL){
        		MPI_Init(NULL, NULL);
        		mpi_comm = MPI_COMM_WORLD;
        		//MPI_Comm_rank(mpi_comm, &rank);
        		//MPI_Comm_size(mpi_comm, &world_size);
        	}
        
            //std::cout << "MPI Comm (" << rank << "," << world_size << ")"<< std::endl;
            assert(world_size>0);
            assert(rank>=0);
        	count_comm+=1;
            return std::make_unique< Comm<DEVICETYPE::MPI> >( (int) rank, (int) world_size );
        };

		MPICommInp(std::array<int,2> nprow):nprow(nprow){};
	//private:
		std::array<int,2> nprow;
};

template<>
Comm<DEVICETYPE::MPI>::~Comm(){
    //std::cout << "comm count: " << count << std::endl;
    int ictxt;
    count_comm-=1;
    if (count_comm==0 && mpi_comm!=MPI_COMM_NULL){
        int status = MPI_Finalize();
        assert (status == MPI_SUCCESS);
        mpi_comm=MPI_COMM_NULL;
		blacs_exit(&ictxt);
    }
    
}

template<>
void Comm<DEVICETYPE::MPI>::finalize(){
//    int status = MPI_Finalize();
//    assert (status == MPI_SUCCESS);
}

template<>
void Comm<DEVICETYPE::MPI>::barrier() const{
    MPI_Barrier(mpi_comm);
}

template <> 
template <> 
void Comm<DEVICETYPE::MPI>::allreduce(const double *src, int count, double *trg, OPTYPE op) const{
    switch (op){
        case OPTYPE::SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  mpi_comm); break;
        case OPTYPE::PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, mpi_comm); break;
        case OPTYPE::MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  mpi_comm); break;
        case OPTYPE::MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  mpi_comm); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}
template <> 
template <> 
void Comm<DEVICETYPE::MPI>::alltoall(double *src, int sendcount, double *trg, int recvcount) const{
    MPI_Alltoall(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<DEVICETYPE::MPI>::alltoallv(double *src, int* sendcounts, double *trg, int* recvcounts) const{
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
void Comm<DEVICETYPE::MPI>::allgather(double *src, int sendcount, double *trg, int recvcount) const{
    MPI_Allgather(src, (int)sendcount, MPI_DOUBLE, trg, (int)recvcount, MPI_DOUBLE, mpi_comm);
}

template <> 
template <> 
void Comm<DEVICETYPE::MPI>::allgatherv(double *src, int sendcount, double *trg, int* recvcounts) const{
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
void Comm<DEVICETYPE::MPI>::scatterv(double *src, int* sendcounts, double *trg, int recvcount, int root) const{
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
