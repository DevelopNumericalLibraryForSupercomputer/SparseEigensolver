//#include "Matrix.hpp"
//#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include <iostream>
#include "ContiguousMap.hpp"
#include <iomanip>
#include "DenseTensor.hpp"

std::ostream& operator<<(std::ostream& os, std::array<size_t,3> &A){
    os << "(";
    for(int i = 0; i<3; i++){
        os << A[i] << " ";
    }
    os << ")";
    return os;
}
using namespace SE;

int main(int argc, char* argv[]){
    Comm<computEnv::MPI> comm;
    comm.initialize(argc, argv);
    std::cout << "MPI test" << std::endl;
    //Comm<computEnv::MKL> comm;
    double x = 0.0, sum = 0.0;
    int myrank = comm.rank;
    int nprocs = comm.world_size;
        
    int nn=100000;
    double step = 0.00001;
    
    int myinit = myrank*(nn/nprocs);
    int myfin =  (myrank+1)*(nn/nprocs)-1;
    if(myfin > nn) myfin = nn;
    if(myrank == 0) { std::cout << "allreduce test" << std::endl;}
    comm.barrier();
    std::cout << "myrank : " << myrank << ", myinit : " << myinit << ", myfin : " << myfin << std::endl;
    for(int i = myinit ; i<=myfin ; i++){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    double tsum = 0.0;
    comm.barrier();
    comm.allreduce(&sum,1,&tsum,SE::SUM);
    //MPI_Allreduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
    sum = tsum = 0;
    
    comm.barrier();
    for(int i = myrank ; i<nn ;i=i+nprocs){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    comm.allreduce(&sum,1,&tsum,SE::SUM);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;

    if(myrank == 0) { std::cout << "alltoall test" << std::endl;}
    double* irecv = (double *)malloc(sizeof(double)*100);
    double* test_array = (double *)malloc(sizeof(double)*100);
    for(int i=0;i<100;i++){
        test_array[i] = (double)i*(myrank+1);
    }
    comm.alltoall(test_array,100/nprocs,irecv,100/nprocs);    
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
    comm.barrier();
    if(myrank == 1) { std::cout << "allgather test" << std::endl;}

    comm.allgather(test_array,100/nprocs,irecv,100/nprocs);
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
    comm.barrier();
    
    std::cout << "ContiguousMap test" << std::endl;
    
    std::array<size_t,3> shape3 = {8,7,17}; 
    
    ContiguousMap<3> cont_map = ContiguousMap<3>(shape3);
    //nproc = 3
    std::array<size_t,3> test_index1 = {1,3,14}; // 1+ 3*8 + 14*8*7 = 809
    size_t test_index1_ = 809;
    if(comm.rank == 0){
        int slice_dimension = comm.rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm.rank, comm.world_size); // 1+ 3*2 + 14*2*7 = 203
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm.rank, comm.world_size); // 1+ 3*2 + 14*2*7 = 203
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (1 3 14 ) = 203std::endl; // rank 0 : (1 3 14 ) = 203
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm.rank == 1){
        int slice_dimension = comm.rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm.rank, comm.world_size); // 1+ 1*8 + 14*8*2 = 233
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm.rank, comm.world_size); // 1+ 1*8 + 14*8*2 = 233
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (1 1 14 ) = 233
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm.rank == 2){
        int slice_dimension = comm.rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm.rank, comm.world_size); // 1+ 3*8 + 4*8*7 = 249
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm.rank, comm.world_size); // 1+ 3*8 + 4*8*7 = 249
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (1 3 4 ) = 249
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm.barrier();

    std::array<size_t,3> test_index2 = {7,6,16}; // 7+ 6*8 + 16*8*7 = 951
    size_t test_index2_ = 951;
    if(comm.rank == 2){
        ////////////////////////////////////
        int slice_dimension = 0;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index2, slice_dimension, comm.rank, comm.world_size); // 3+ 6*4 + 16*4*7 = 475
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm.rank, comm.world_size); // 3+ 6*4 + 16*4*7 = 475
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (3 6 16 ) = 475
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 1;
        //array G -> array L
        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        local = cont_map.get_local_index(test_index2, slice_dimension, comm.rank, comm.world_size); // 7+ 2*8 + 16*8*3 = 407
        //size_t G -> array L
        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm.rank, comm.world_size); // 7+ 2*8 + 16*8*3 = 407
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (7 2 16 ) = 407
        //array L -> array G
        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 2;
        //array G -> array L
        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm.rank, comm.world_size);
        //array G -> size_t L
        local = cont_map.get_local_index(test_index2, slice_dimension, comm.rank, comm.world_size); // 7+ 6*8 + 6*8*7 = 391
        //size_t G -> array L
        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm.rank, comm.world_size);
        //size_t G -> size_t L
        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm.rank, comm.world_size); // 7+ 6*8 + 6*8*7 = 391
        std::cout << "rank " << comm.rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (7 6 6 ) = 391
        //array L -> array G
        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> array G
        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm.rank, comm.world_size);
        //array L -> size_t G
        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm.rank, comm.world_size);
        //size_t L -> size_t G
        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm.rank, comm.world_size);
        std::cout << "rank " << comm.rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm.barrier();
    
    std::array<size_t, 2> test_shape = {3,3};
    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    

    ContiguousMap<2> new_map = ContiguousMap<2>(test_shape);

    SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > test_matrix
                = SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > (test_shape, &test_data[0]);
    //test_matrix.decompose("EVD");
    comm.barrier();
    //MPI_Finalize();
    return 0;
}
