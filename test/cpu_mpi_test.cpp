#include <vector>
#include <array>
#include <iostream>
#include <iomanip>


#include "device/MPI/LinearOp.hpp"
#include "device/MPI/TensorOp.hpp"
#include "device/MPI/MPIComm.hpp"
#include "ContiguousMap.hpp"
#include "decomposition/IterativeSolver_MPI.hpp"
#include "decomposition/Decompose.hpp"

#include <chrono>

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
    auto comm = createComm<MPI>(argc, argv);
    std::cout << "MPI test" << std::endl;
    double x = 0.0, sum = 0.0;
    int myrank = comm->rank;
    int nprocs = comm->world_size;
    /*      
    int nn=100000;
    double step = 0.00001;

    int myinit = myrank*(nn/nprocs);
    int myfin =  (myrank+1)*(nn/nprocs)-1;
    if(myfin > nn) myfin = nn;
    if(myrank == 0) { std::cout << "allreduce test" << std::endl;}
    comm->barrier();
    std::cout << "myrank : " << myrank << ", myinit : " << myinit << ", myfin : " << myfin << std::endl;
    for(int i = myinit ; i<=myfin ; i++){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    double tsum = 0.0;
    comm->barrier();
    comm->allreduce<double>(&sum,1,&tsum,SEop::SUM);
    //MPI_Allreduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
    sum = tsum = 0;
    
    comm->barrier();
    for(int i = myrank ; i<nn ;i=i+nprocs){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    comm->allreduce(&sum,1,&tsum,SEop::SUM);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;

    if(myrank == 0) { std::cout << "alltoall test" << std::endl;}
    double* irecv = (double *)malloc(sizeof(double)*100);
    double* test_array = (double *)malloc(sizeof(double)*100);
    for(int i=0;i<100;i++){
        test_array[i] = (double)i*(myrank+1);
    }
    comm->alltoall(test_array,100/nprocs,irecv,100/nprocs);    
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
    comm->barrier();
    if(myrank == 1) { std::cout << "allgather test" << std::endl;}

    comm->allgather(test_array,100/nprocs,irecv,100/nprocs);
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
    comm->barrier();
    */
    /*
    std::cout << "ContiguousMap test" << std::endl;
    
    std::array<size_t,3> shape3 = {8,7,17}; 
    
    ContiguousMap<3> cont_map = ContiguousMap<3>(shape3);
    //nproc = 3
    std::array<size_t,3> test_index1 = {1,3,14}; // 1+ 3*8 + 14*8*7 = 809
    size_t test_index1_ = 809;
    if(comm->rank == 0){
        int slice_dimension = comm->rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (1 3 14 ) = 203std::endl; // rank 0 : (1 3 14 ) = 203
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm->rank == 1){
        int slice_dimension = comm->rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (1 1 14 ) = 233
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm->rank == 2){
        int slice_dimension = comm->rank;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (1 3 4 ) = 249
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm->barrier();

    std::array<size_t,3> test_index2 = {7,6,16}; // 7+ 6*8 + 16*8*7 = 951
    size_t test_index2_ = 951;
    if(comm->rank == 2){
        ////////////////////////////////////
        int slice_dimension = 0;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        size_t local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        size_t local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (3 6 16 ) = 475
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        size_t restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        size_t restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 1;
        //array G -> array L
        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
        //size_t G -> array L
        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (7 2 16 ) = 407
        //array L -> array G
        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 2;
        //array G -> array L
        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
        //array G -> size_t L
        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
        //size_t G -> array L
        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
        //size_t G -> size_t L
        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (7 6 6 ) = 391
        //array L -> array G
        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> array G
        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
        //array L -> size_t G
        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
        //size_t L -> size_t G
        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm->barrier();
    */
    /*
    std::array<size_t, 2> test_shape = {3,3};
    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    ContiguousMap<2> new_map = ContiguousMap<2>(test_shape);
    SE::DenseTensor<double, 2, Comm<MPI>, ContiguousMap<2> > test_matrix
                = SE::DenseTensor<double, 2, Comm<MPI>, ContiguousMap<2> > (test_shape, &test_data[0]);
    auto out = test_matrix.decompose("EVD");
    if (comm->rank == 0){
        print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());
    }
    comm->barrier();
    */
    std::cout << "Sparse Matrix test" << std::endl;

    size_t N = 30;
    std::array<size_t, 2> test_shape2 = {N,N};
    ContiguousMap<2>* new_map2 = new ContiguousMap<2>(test_shape2, comm.get()->world_size, 0);
    SE::Tensor<SE::STORETYPE::COO, double, 2, MPI, ContiguousMap<2> > test_sparse(comm.get(), new_map2, test_shape2);

    
    std::cout << "Matrix construction" << std::endl;
    size_t chunk_size = test_sparse.shape[0] / test_sparse.comm->world_size;
    size_t* local_matrix_size = new_map2->get_partition_size_array();

    for(size_t i=0;i<N;i++){
    //for(size_t loc_i=0; loc_i<local_matrix_size[myrank]; loc_i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(new_map2->get_my_rank_from_global_index(index)==myrank){
                if(i == j)                   test_sparse.insert_value(index, 2.0*((double)i+1.0-(double)N));
                //if(i == j +1 || i == j -1)   test_sparse.insert_value(index, 3.0);
                if(i == j +2 || i == j -2)   test_sparse.insert_value(index, -1.0);
                if(i == j +3 || i == j -3)   test_sparse.insert_value(index, 0.3);
                //if(i == j +4 || i == j -4)   test_sparse.insert_value(index, -0.1);
            }
            else{
                continue;
            }
        }
    }
    test_sparse.complete();
    std::cout << "matrix construction complete" << std::endl;
    test_sparse.print();
    /*
    comm->barrier();
    if (comm->rank == 0) std::cout << "Sparsematrix Davidson" << std::endl;
    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();  
    auto out3 = decompose(test_sparse, "davidson");
    if (comm->rank == 0) print_eigenvalues( "Eigenvalues", 3, out3.get()->real_eigvals.get(), out3.get()->imag_eigvals.get());
    comm->barrier();
    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
    if (comm->rank == 0) std::cout << "BlockDavidson_sparse, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count())/1000000.0 << "[sec]" << std::endl;
   */




    return 0;
}
