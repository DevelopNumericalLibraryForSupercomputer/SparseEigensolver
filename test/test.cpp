#include "Matrix.hpp"
#include "DenseTensor.hpp"
#include <vector>
#include <array>
#include <iostream>
#include "Device.hpp"
#include "ContiguousMap.hpp"
#include "Utility_include.hpp"
#include <iomanip>
#include "Comm.hpp"
#include "device/CPU/Comm.hpp"

std::ostream& operator<<(std::ostream& os, std::array<size_t,3> &A){
    os << "(";
    for(int i = 0; i<3; i++){
        os << A[i] << " ";
    }
    os << ")";
    return os;
}

int main(int argc, char* argv[]){
    TH::Comm<TH::CPU> comm = TH::Comm<TH::CPU>(argc, argv);
    
    comm.barrier();
    std::cout << "MPI test" << std::endl;

    double x = 0.0, sum = 0.0;
    int myrank = comm.get_rank();
    int nprocs = comm.get_world_size();
        
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
    comm.allreduce(&sum,1,&tsum,TH::SUM);
    std::cout.precision(10);
    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
    sum = tsum = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    for(int i = myrank ; i<nn ;i=i+nprocs){
        x = ((double)i+0.5)*step;
        sum = sum + 4.0/(1.0+x*x);
    }
    comm.allreduce(&sum,1,&tsum,TH::SUM);
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
    /*
    //if(comm.get_rank() == 0){
        std::cout << "Matrix muliplication test" << std::endl;
        size_t m, k, n;
        m = 4;
        k = 2;
        n = 3;
        std::vector<double> A_vec = {1,2,3,4,5,6,7,8};
        std::vector<double> B_vec = {1,2,3,4,5,6};
        std::vector<double> AT_vec = {1,5,2,6,3,7,4,8};
        std::vector<double> BT_vec = {1,3,5,2,4,6};
        std::cout << m << " " << k << " " << n << std::endl;

        TH::Matrix<double> Amat = TH::Matrix<double>(m,k,A_vec.data());
        TH::Matrix<double> Bmat = TH::Matrix<double>(k,n,B_vec.data());
        TH::Matrix<double> Cmat = TH::Matrix<double>(m,n);

        TH::Matrix<double> AmatT = TH::Matrix<double>(k,m,AT_vec.data());
        TH::Matrix<double> BmatT = TH::Matrix<double>(n,k,BT_vec.data());


        Cmat.multiply(Amat,Bmat,"NoT","NoT");
        std::cout << Amat << Bmat << Cmat << std::endl;

        Cmat.multiply(Amat,BmatT,"NoT","T");
        std::cout << Amat << BmatT << Cmat << std::endl;

        Cmat.multiply(AmatT,Bmat,"T","NoT");
        std::cout << AmatT << Bmat << Cmat << std::endl;

        Cmat.multiply(AmatT,BmatT,"T","T");
        std::cout << AmatT << BmatT << Cmat << std::endl;

        std::cout << "DenseTensor Operator test" << std::endl;
        std::array<size_t,4> shape = {1,3,2,4};
        TH::DenseTensor<double,4,TH::CPU,TH::Comm<TH::CPU>> Aten = TH::DenseTensor<double,4,TH::CPU,TH::Comm<TH::CPU>>(shape);
    
        std::vector<double> Bten_vec = {};

        std::cout << 1*3*2*4 << " : ";
        for(size_t i=0; i<1*3*2*4; ++i){
            Aten[i]=1.0;
            std::cout << Aten[i] << " ";
            Bten_vec.push_back((double)i);
        }
        std::cout << std::endl;
        std::cout << "shape = (" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "), shape_mult : ";

        for(int i=0;i<Aten.shape_mult.size();++i){
            std::cout << Aten.shape_mult[i] << " ";
        }
        std::cout << std::endl;
        TH::DenseTensor<double,4,TH::CPU,TH::Comm<TH::CPU>> Bten = TH::DenseTensor<double,4,TH::CPU,TH::Comm<TH::CPU>>(shape,Bten_vec.data());
        std::array<size_t,4> index_array = {0,1,1,2};
        std::cout << "0,1,1,2 : " << Bten(index_array) << std::endl;
        index_array[3]=0;
        std::cout << "0,1,1,0 : " << Bten(index_array) << std::endl;
        index_array[1]=0;
        Bten.insert_value(index_array,-3.2);
        std::cout << "0,0,1,0 : " << Bten(index_array) << std::endl;
        Bten[8]=3.14;
        for(size_t i=0; i<Bten.shape_mult[4]; ++i){
            std::cout << Bten[i] << " " ;
        }
        std::cout << std::endl;
        TH::DenseTensor<double,4,TH::CPU,TH::Comm<TH::CPU>> Cten = Bten.clone();
        for(size_t i=0; i<Cten.shape_mult[4]; ++i){
            std::cout << Cten[i] << " " ;
        }
        std::cout << std::endl;
        Cten = Aten;
        for(size_t i=0; i<Cten.shape_mult[4]; ++i){
            std::cout << Cten[i] << " " ;
        }
        Aten = Cten.clone();
        for(size_t i=0; i<Aten.shape_mult[4]; ++i){
            std::cout << Aten[i] << " " ;
        }
        std::cout << std::endl;

    //}
    */
    std::cout << "ContiguousMap test" << std::endl;
    
    std::array<size_t,3> shape3 = {8,7,17}; 
    
    TH::Map<3,TH::CPU>* cont_map = new TH::ContiguousMap<3,TH::CPU>(shape3, comm);
    //nproc = 3
    std::array<size_t,3> test_index1 = {1,3,14}; // 1+ 3*8 + 14*8*7 = 809
    size_t test_index1_ = 809;
    if(comm.get_rank() == 0){
        int slice_dimension = comm.get_rank();
        //array G -> array L
        std::array<size_t,3> sliced = cont_map->get_local_array_index(test_index1, slice_dimension);
        //array G -> size_t L
        size_t local = cont_map->get_local_index(test_index1, slice_dimension); // 1+ 3*2 + 14*2*7 = 203
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map->get_local_array_index(test_index1_, slice_dimension);
        //size_t G -> size_t L
        size_t local_ = cont_map->get_local_index(test_index1_, slice_dimension); // 1+ 3*2 + 14*2*7 = 203
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (1 3 14 ) = 203std::endl; // rank 0 : (1 3 14 ) = 203
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        size_t restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        size_t restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm.get_rank() == 1){
        int slice_dimension = comm.get_rank();
        //array G -> array L
        std::array<size_t,3> sliced = cont_map->get_local_array_index(test_index1, slice_dimension);
        //array G -> size_t L
        size_t local = cont_map->get_local_index(test_index1, slice_dimension); // 1+ 1*8 + 14*8*2 = 233
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map->get_local_array_index(test_index1_, slice_dimension);
        //size_t G -> size_t L
        size_t local_ = cont_map->get_local_index(test_index1_, slice_dimension); // 1+ 1*8 + 14*8*2 = 233
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (1 1 14 ) = 233
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        size_t restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        size_t restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    if(comm.get_rank() == 2){
        int slice_dimension = comm.get_rank();
        //array G -> array L
        std::array<size_t,3> sliced = cont_map->get_local_array_index(test_index1, slice_dimension);
        //array G -> size_t L
        size_t local = cont_map->get_local_index(test_index1, slice_dimension); // 1+ 3*8 + 4*8*7 = 249
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map->get_local_array_index(test_index1_, slice_dimension);
        //size_t G -> size_t L
        size_t local_ = cont_map->get_local_index(test_index1_, slice_dimension); // 1+ 3*8 + 4*8*7 = 249
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (1 3 4 ) = 249
        
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        size_t restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        size_t restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm.barrier();

    std::array<size_t,3> test_index2 = {7,6,16}; // 7+ 6*8 + 16*8*7 = 951
    size_t test_index2_ = 951;
    if(comm.get_rank() == 2){
        ////////////////////////////////////
        int slice_dimension = 0;
        //array G -> array L
        std::array<size_t,3> sliced = cont_map->get_local_array_index(test_index2, slice_dimension);
        //array G -> size_t L
        size_t local = cont_map->get_local_index(test_index2, slice_dimension); // 3+ 6*4 + 16*4*7 = 475
        //size_t G -> array L
        std::array<size_t,3> sliced_ = cont_map->get_local_array_index(test_index2_, slice_dimension);
        //size_t G -> size_t L
        size_t local_ = cont_map->get_local_index(test_index2_, slice_dimension); // 3+ 6*4 + 16*4*7 = 475
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (3 6 16 ) = 475
        //array L -> array G
        std::array<size_t, 3> restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        std::array<size_t, 3> restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        size_t restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        size_t restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 1;
        //array G -> array L
        sliced = cont_map->get_local_array_index(test_index2, slice_dimension);
        //array G -> size_t L
        local = cont_map->get_local_index(test_index2, slice_dimension); // 7+ 2*8 + 16*8*3 = 407
        //size_t G -> array L
        sliced_ = cont_map->get_local_array_index(test_index2_, slice_dimension);
        //size_t G -> size_t L
        local_ = cont_map->get_local_index(test_index2_, slice_dimension); // 7+ 2*8 + 16*8*3 = 407
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (7 2 16 ) = 407
        //array L -> array G
        restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
        ////////////////////////////////////
        slice_dimension = 2;
        //array G -> array L
        sliced = cont_map->get_local_array_index(test_index2, slice_dimension);
        //array G -> size_t L
        local = cont_map->get_local_index(test_index2, slice_dimension); // 7+ 6*8 + 6*8*7 = 391
        //size_t G -> array L
        sliced_ = cont_map->get_local_array_index(test_index2_, slice_dimension);
        //size_t G -> size_t L
        local_ = cont_map->get_local_index(test_index2_, slice_dimension); // 7+ 6*8 + 6*8*7 = 391
        std::cout << "rank " << comm.get_rank() << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (7 6 6 ) = 391
        //array L -> array G
        restored1 = cont_map->get_global_array_index(sliced, slice_dimension);
        //size_t L -> array G
        restored2 = cont_map->get_global_array_index(local, slice_dimension);
        //array L -> size_t G
        restored1_ = cont_map->get_global_index(sliced_, slice_dimension);
        //size_t L -> size_t G
        restored2_ = cont_map->get_global_index(local_, slice_dimension);
        std::cout << "rank " << comm.get_rank() << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
    }
    comm.barrier();

    
    //std::cout << "after barrier" << std::endl;
        
    return 0;
}
