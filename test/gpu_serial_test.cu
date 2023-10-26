//#include "Matrix.hpp"
//#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include "device/CUDA/CUDAComm.hpp"
#include "device/CUDA/Utility.hpp"
#include "ContiguousMap.hpp"
//#include <iomanip>
#include "DenseTensor.hpp"
//#include "decomposition/DirectSolver.hpp"
using namespace std;
using namespace SE;

__global__ void kernel_sum2(int myinit, int myfin, double step, double* sum){
    // This is super slow only for test!
    double x_sum=0;
    for (int i=myinit+blockDim.x*blockIdx.x+threadIdx.x; i<myfin; i+=gridDim.x*blockDim.x){
        double x = ((double)i+0.5)*step;
        x_sum += 4.0 / (1.0+ x*x);
    }
    atomicAdd(sum, x_sum ); // sm_60 or higher
    return;
}

#define env computEnv::CUDA
int main(int argc, char* argv[]){
    auto comm = createComm<env>(argc, argv);

    
    auto host_sum  = malloc<double> (1);
    auto host_tsum = malloc<double> (1);

    std::cout << "gpu SERIAL test" << std::endl;
    std::cout << *comm <<std::endl;
    double* sum  = malloc<double, env>(1);
    memset<double, env>(sum, 0, 1);
    double* tsum = malloc<double, env>(1);
    memset<double, env>(tsum, 0, 1);
    int myrank = comm->rank;
    int nprocs = comm->world_size;
        
    int nn=100000;
    double step = 0.00001;
    
    int myinit = myrank*(nn/nprocs);
    int myfin =  (myrank+1)*(nn/nprocs)-1;
    if(myfin > nn) myfin = nn;
    if(myrank == 0) { std::cout << "allreduce test" << std::endl;}
    comm->barrier();

    std::cout << "myrank : " << myrank << ", myinit : " << myinit << ", myfin : " << myfin << std::endl;
    kernel_sum2<<< 128,16 >>> (myinit, myfin, step, sum);

    comm->allreduce<double>(sum,1,tsum,SE::SUM);
    std::cout.precision(10);
    cudaMemcpy(host_sum,   sum, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_tsum, tsum, 1*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "myrank : " << myrank << ", sum = " << *host_sum << ", tsum*step = " << (*host_tsum)*step << std::endl;
    memset<double, env>(sum, 0, 1);
    memset<double, env>(tsum, 0, 1);
    
    comm->barrier();
    kernel_sum2<<< 128, 16 >>>(myinit, myfin, step, sum);
    comm->allreduce(sum,1,tsum,SE::SUM);
    std::cout.precision(10);
    cudaMemcpy(host_sum,   sum, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_tsum, tsum, 1*sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "myrank : " << myrank << ", sum = " << *host_sum << ", tsum*step = " << (*host_tsum)*step << std::endl;


    // need to edit!!

    if(myrank == 0) { std::cout << "alltoall test" << std::endl;}
    auto irecv      = malloc<double,env >(sizeof(double)*100);
    auto test_array_ = malloc<double>(100);
    for(int i=0;i<100;i++){
        test_array_[i] = (double)i*(myrank+1);
    }
    auto test_array = malloc<double,env>(100);
    cudaMemcpy(test_array, test_array_, 1*sizeof(double), cudaMemcpyHostToDevice);
    comm->alltoall(test_array,100/nprocs,irecv,100/nprocs);   
    cudaMemcpy(test_array_, test_array, 1*sizeof(double), cudaMemcpyDeviceToHost);
 
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
    comm->barrier();
    if(myrank == 1) { std::cout << "allgather test" << std::endl;}

//    comm->allgather(test_array,100/nprocs,irecv,100/nprocs);
//    if(myrank == 1) {
//        for(int i=0;i<100;i++){
//            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
//        }
//    }
//    comm->barrier();
//    std::cout << "ContiguousMap test" << std::endl;
//    
//    std::array<size_t,3> shape3 = {8,7,17}; 
//    
//    ContiguousMap<3> cont_map = ContiguousMap<3>(shape3);
//    //nproc = 3
//    std::array<size_t,3> test_index1 = {1,3,14}; // 1+ 3*8 + 14*8*7 = 809
//    size_t test_index1_ = 809;
//    if(comm->rank == 0){
//        int slice_dimension = comm->rank;
//        //array G -> array L
//        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
//        //array G -> size_t L
//        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
//        //size_t G -> array L
//        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
//        //size_t G -> size_t L
//        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
//        ////std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (1 3 14 ) = 203std::endl; // rank 0 : (1 3 14 ) = 203
//        
//        //array L -> array G
//        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
//        //size_t L -> array G
//        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
//        //array L -> size_t G
//        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
//        //size_t L -> size_t G
//        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
//        ////std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
//    }
////    if(comm->rank == 1){
////        int slice_dimension = comm->rank;
////        //array G -> array L
////        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
////        //array G -> size_t L
////        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
////        //size_t G -> array L
////        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
////        //size_t G -> size_t L
////        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (1 1 14 ) = 233
////        
////        //array L -> array G
////        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> array G
////        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> size_t G
////        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> size_t G
////        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    if(comm->rank == 2){
////        int slice_dimension = comm->rank;
////        //array G -> array L
////        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
////        //array G -> size_t L
////        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
////        //size_t G -> array L
////        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
////        //size_t G -> size_t L
////        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (1 3 4 ) = 249
////        
////        //array L -> array G
////        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> array G
////        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> size_t G
////        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> size_t G
////        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    comm->barrier();
////
////    std::array<size_t,3> test_index2 = {7,6,16}; // 7+ 6*8 + 16*8*7 = 951
////    size_t test_index2_ = 951;
////    if(comm->rank == 2){
////        ////////////////////////////////////
////        int slice_dimension = 0;
////        //array G -> array L
////        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> size_t L
////        size_t local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
////        //size_t G -> array L
////        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //size_t G -> size_t L
////        size_t local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (3 6 16 ) = 475
////        //array L -> array G
////        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> array G
////        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> size_t G
////        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> size_t G
////        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////        ////////////////////////////////////
////        slice_dimension = 1;
////        //array G -> array L
////        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> size_t L
////        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
////        //size_t G -> array L
////        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //size_t G -> size_t L
////        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (7 2 16 ) = 407
////        //array L -> array G
////        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> array G
////        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> size_t G
////        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> size_t G
////        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////        ////////////////////////////////////
////        slice_dimension = 2;
////        //array G -> array L
////        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> size_t L
////        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
////        //size_t G -> array L
////        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //size_t G -> size_t L
////        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (7 6 6 ) = 391
////        //array L -> array G
////        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> array G
////        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> size_t G
////        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //size_t L -> size_t G
////        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    comm->barrier();
    
    std::array<size_t, 2> test_shape = {3,3};
    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    

    ContiguousMap<2> new_map = ContiguousMap<2>(test_shape);

    SE::DenseTensor<double, 2, Comm<SE::env>, ContiguousMap<2> > test_matrix(test_shape, &test_data[0]);
//                = SE::DenseTensor<double, 2, Comm<SE::PROTOCOL::SERIAL>, ContiguousMap<2> > (test_shape, &test_data[0]);
    comm->barrier();

    //auto out = test_matrix.decompose("EVD");

    //print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());

    //SERIAL_Finalize();
    return 0;
}
