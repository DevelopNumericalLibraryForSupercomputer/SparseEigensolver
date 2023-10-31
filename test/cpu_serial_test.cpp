//#include "Matrix.hpp"
//#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include "device/MKL/MKLComm.hpp"
//#include "ContiguousMap.hpp"
//#include <iomanip>
//#include "DenseTensor.hpp"
#include "decomposition/DirectSolver.hpp"

#include "decomposition/IterativeSolver.hpp"

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
    auto comm = createComm<computEnv::MKL >(argc, argv);
    std::cout << "SERIAL test" << std::endl;
    std::cout << *comm <<std::endl;
    double x = 0.0, sum = 0.0;
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
    for(int i = myinit ; i<=myfin ; i++){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    double tsum = 0.0;
    comm->allreduce<double>(&sum,1,&tsum,SE::SUM);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
    sum = tsum = 0;
    
    comm->barrier();
    for(int i = myrank ; i<nn ;i=i+nprocs){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    comm->allreduce(&sum,1,&tsum,SE::SUM);
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
    if(myrank == 1) { std::cout << "allgather test" << std::endl;}

    comm->allgather(test_array,100/nprocs,irecv,100/nprocs);
    if(myrank == 1) {
        for(int i=0;i<100;i++){
            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
        }
    }
        
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
    
    std::array<size_t, 2> test_shape = {3,3};
    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    

    ContiguousMap<2> new_map = ContiguousMap<2>(test_shape);

    SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > test_matrix(test_shape, &test_data[0]);
//                = SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > (test_shape, &test_data[0]);
    comm->barrier();

    auto out = test_matrix.decompose("EVD");

    print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());

    size_t N = 100;
    std::array<size_t, 2> test_shape2 = {N,N};
    double* test_data2 = malloc<double, computEnv::MKL>(N*N);
    for(int i=0;i<N*N;i++){
        test_data2[i] = 0;
        if(i%(N+1)==0){
            test_data2[i]  = -((double)N) + (double)(i/(N+1) + 1);
            if (i-1 >0) test_data2[i-1] += 3.0;
            if (i-2 >0) test_data2[i-2] += -1.0;
            if (i-3 >0) test_data2[i-3] += 0.3;
            if (i-4 >0) test_data2[i-4] += -0.1;
        }
        else if(i%7==0){
            test_data2[i] += 0.1; //(i%N, i/N) i%N + i/N * N 
        }
    }
    for(int i=0;i<N;i++){
        for(int j=i;j<N;j++){
            test_data2[j*N+i] = test_data2[j*N+i] + test_data2[i+j*N];
            test_data2[i*N+j] = test_data2[j*N+i];
        }
    }
    /*
    std::cout << "matrix!" << std::endl;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            std::cout << std::setw(6) << test_data2[i+j*N] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "matrix!" << std::endl;
    */
    ContiguousMap<2> new_map2 = ContiguousMap<2>(test_shape2);
    SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > test_matrix2(test_shape2, test_data2);
//                = SE::DenseTensor<double, 2, Comm<SE::computEnv::MKL>, ContiguousMap<2> > (test_shape, &test_data[0]);
    
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = test_matrix2.decompose("EVD");
    print_eigenvalues( "Eigenvalues", 10, out1.get()->real_eigvals.get(), out1.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    std::cout << "Time difference = " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;

    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
    auto out2 = test_matrix2.decompose("Davidson");
    print_eigenvalues( "Eigenvalues", 10, out2.get()->real_eigvals.get(), out2.get()->imag_eigvals.get());
    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    
    std::cout << "Time difference = " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;
    
    return 0;
}
