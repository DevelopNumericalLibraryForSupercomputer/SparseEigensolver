
#include <iostream>
#include <string>
#include <utility>
#include <memory>
#include <cstring>
#include <cassert>
#include "mkl.h"
#include "mpi.h"


struct ComputEnv{
    const std::string env_name;
    constexpr ComputEnv(): env_name("BASE"){};
};
struct MKL: public ComputEnv{
    const std::string env_name = "MKL";
};
struct MPI: public ComputEnv{
    const std::string env_name = "MPI";
};

enum class SEop{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
};

template<typename env>
class Comm{
public:
    Comm(size_t rank, size_t world_size): rank(rank), world_size(world_size) {};
    Comm(){};
    ~Comm(){};
    int rank;
    int world_size;

    void barrier() {};
    template <typename datatype> void allreduce(const datatype *src, size_t count, datatype *trg, SEop op);
};
template<typename env>
std::unique_ptr<Comm<env> > createComm(int argc, char *argv[]){
    std::cout << "empty comm" << std::endl;
    return std::make_unique< Comm<env> >();
};

template<>
std::unique_ptr<Comm<MKL> > createComm(int argc, char *argv[]){
    std::cout << "SERIALcomm" << std::endl;
    return std::make_unique< Comm<MKL> >( 0, 1 );
}
template<>
template<typename datatype>
void Comm<MKL>::allreduce(const datatype *src, size_t count, datatype *trg, SEop op){
    std::memcpy(trg, src, count*sizeof(datatype));
}

template<>
std::unique_ptr<Comm<MPI> > createComm(int argc, char *argv[]){
    std::cout << "MPIcomm" << std::endl;
    MPI_Init(&argc, &argv);
    int myRank ,nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);
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
template <> 
template <> 
void Comm<MPI>::allreduce(const double *src, size_t count, double *trg, SEop op){
    switch (op){
        case SEop::SUM:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD); break;
        case SEop::PROD: MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_PROD, MPI_COMM_WORLD); break;
        case SEop::MAX:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD); break;
        case SEop::MIN:  MPI_Allreduce(src, trg, count, MPI_DOUBLE, MPI_MIN,  MPI_COMM_WORLD); break;
        default: std::cout << "WRONG OPERATION TYPE" << std::endl; exit(-1);
    }
}
int main(int argc, char* argv[]){
    auto comm = createComm<MPI>(argc, argv);
    std::cout << comm.get()->rank << " / " << comm.get()->world_size << std::endl;
    
    return 0;
}