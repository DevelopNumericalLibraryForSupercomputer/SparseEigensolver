// communicator
// 참고: https://sooftware.io/big-model2/
// Chat-GPT 3.5
#include <iostream>
#include <string>
#include <mpi.h>
//#include <nccl.h>

namespace TensorHetero{
//template <typename datatype>
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

    const size_t get_rank();
    const size_t get_local_rank();
    const size_t get_world_size();
    const std::string get_comm_protocol();

    //void allreduce<typename datatype> (datatype* src, Operator);       //int MPI_Allreduce(void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, MPI_Comm comm)
    //void all2all  <typename datatype> (datatype* src, datatype* trg);  //int MPI_Alltoall (void *sendbuf, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
    //void allgather<typename datatype> (datatype* src, datatype* trg);  //int MPI_Allgather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm)
};

Comm::Comm(const std::string &protocol){
    if (protocol == "mpi"){
        // Initialize MPI (assuming MPI_Init has been called before this)
        int tmp_rank, tmp_world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
        rank = tmp_rank;
        world_size = tmp_world_size;
        comm_protocol = protocol;
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
Comm::Comm(int argc, char *argv[], const std::string &protocol){
    if (protocol == "mpi"){
        // Initialize MPI (assuming MPI_Init has been called before this)
        int tmp_rank, tmp_world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &tmp_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &tmp_world_size);
        rank = tmp_rank;
        world_size = tmp_world_size;
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
    // Add more protocol-specific finalization here if needed
}

inline const size_t Comm::get_rank(){ return rank; }
inline const size_t Comm::get_local_rank(){ return local_rank; }
inline const size_t Comm::get_world_size(){ return world_size; }
inline const std::string Comm::get_comm_protocol(){ return std::string(); }
/*
template <typename datatype>
void Comm::allreduce(datatype *src, Operator){
}
template <typename datatype>
void Comm::all2all(datatype *src, datatype *trg){
}
template <typename datatype>
void Comm::allgather(datatype *src, datatype *trg){
}
*/
}