#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <string>
#include <chrono>
#include <random>

#include "device/mpi/TensorOp.hpp"
#include "device/mpi/LinearOp.hpp"
#include "device/mpi/MPIComm.hpp"
#include "BlockCyclingMap.hpp"
//#include "decomposition/IterativeSolver_MPI.hpp"
#include "decomposition/Decompose.hpp"
#include "decomposition/DecomposeOption.hpp"
#include "Utility.hpp"
using namespace SE;

// predefined const variables 
const int i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;

// print funciton 
std::ostream& operator<<(std::ostream& os, std::array<int,3> &A){
    os << "(";
    for(int i = 0; i<3; i++){
        os << A[i] << " ";
    }
    os << ")";
    return os;
}


int main(int argc, char** argv){
	// predefined value
    int rank=0, nprocs=1, ictxt;
	DecomposeOption option;
	
	// input 
	int p=1;
	if (argc>=2 ) p = std::stoi( argv[1] ); 
	
	int q=1;
	if (argc>=3 ) q = std::stoi( argv[2] );
    
	int N = 6000;
	if (argc>=4 ) N=std::stoi(argv[3]);

	int nb = 2;
	if (argc>=5 ) nb=std::stoi(argv[4]);
	//int p=1; int q=1;
	int precond_type = 2;
	if (argc>=6 ) option.preconditioner = (PRECOND_TYPE) std::stoi(argv[5]);

	// num_eig variable is not used 
    const int num_eig = 3;

	MPICommInp comm_inp({p,q});
    auto ptr_comm = comm_inp.create_comm();
	auto ptr_comm2 = ptr_comm->clone();
	std::cout << "clone!" <<std::endl;

//	std::cout << "nprow and npcol: " << p <<" " << q <<std::endl;
//	#pragma omp parallel
//	{
//		
//		#pragma omp single nowait
//		{
//		std::cout <<"omp_get_num_threads:"<< omp_get_num_threads() <<std::endl;	
//		}
//	}


    // Set the seed for random number generation
    std::mt19937 rng(12345);  // Fixed seed number for reproducibility
    std::uniform_real_distribution<double> dist(-1, 1);  // Define the range of random values

    if(ptr_comm->get_rank()==0) std::cout << "========================\nDense matrix davidson test" << std::endl;



	BlockCyclingMapInp<2> map_inp({N,N}, ptr_comm->get_rank(), ptr_comm->get_world_size(), {nb, nb}, comm_inp.nprow );
	DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> test_matrix2(comm_inp.create_comm(), map_inp.create_map());

    double invh2 = 1.0;
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
			std::array<int,2> global_arr_index={i,j};
            //test_matrix2.global_set_value(global_arr_index, 0.0);
            if(i == j)  test_matrix2.global_insert_value(global_arr_index, 2.0*((double)i-(double)N)   - invh2*5.0/2.0);
            if(i == j +1 || i == j -1) test_matrix2.global_insert_value(global_arr_index, invh2*4.0/3.0);
            if(i == j +2 || i == j -2)  test_matrix2.global_insert_value(global_arr_index, -invh2*1.0/12.0);
            if(i == j +3 || i == j -3)  test_matrix2.global_insert_value(global_arr_index,   0.3);
            if(i == j +4 || i == j -4)  test_matrix2.global_insert_value(global_arr_index, -0.1);
            //if( i%13 == 0 && j%13 == 0) test_data2[i+j*N] += 0.01;
            //if(i>=3 || j>=3) test_data2[i+j*N] = 0.0;
        }
    }
    
	map_inp.global_shape = {N,num_eig};
    auto guess = new DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>(comm_inp.create_comm(), map_inp.create_map());

    // guess : unit vector
    for(int i=0;i<num_eig;i++){
		for (int j=0; j<N; j++){
	        std::array<int, 2> tmp_index = {j,i};
			//if (i==j)	guess->global_set_value(tmp_index, 1.0);
			//else	guess->global_set_value(tmp_index, 1e-3);
			guess->global_set_value(tmp_index, dist(rng));
		}
    }



    if(ptr_comm->get_rank()==0) std::cout << "========================\nDense matrix diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix2, guess, option);
    if(ptr_comm->get_rank()==0) print_eigenvalues( "Eigenvalues", num_eig, out1.get()->real_eigvals.data(), out1.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "block davidson calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    delete guess;
  return 0;
}

//int main(int argc, char* argv[]){
//    auto ptr_comm = create_comm<DEVICETYPE::MPI>(argc, argv);
//    std::cout << "MPI test" << std::endl;
//    std::cout << *ptr_comm <<std::endl;    
//    
//    //std::array<int, 2> test_shape2 = {30,30};
//    //std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
//    const int N = 7;
//    Contiguous1DMap map (std::array<int,2>({N,N}),  ptr_comm->get_rank(), ptr_comm->get_world_size() );
//    //std::cout << "0-" << ptr_comm->get_rank() <<std::endl;
//    SE::DenseTensor<2,double,Contiguous1DMap<2>, DEVICETYPE::MPI> test_matrix2(*ptr_comm,map);
//
//    //test_matrix.complete();
//    //test_matrix.print_tensor();
//    ptr_comm->barrier();
//    for (int rank=0; rank<ptr_comm->get_world_size(); rank++){
//        ptr_comm->barrier();
//        if(rank==ptr_comm->get_rank()){
//            for (int local_i = 0; local_i < test_matrix2.map.get_local_shape(0); local_i++){
//                for (int local_j=0; local_j < test_matrix2.map.get_local_shape(1); local_j++) {
//                    auto global_index = test_matrix2.map.local_to_global(std::array<int,2>({local_i, local_j} ) );
//                    auto i = global_index[0];
//                    auto j = global_index[1];
//                    if(i == j)                 test_matrix2.global_set_value( global_index,  2.0*((double)i+1.0-(double)N) );
//                    if(i == j +2 || i == j -2) test_matrix2.global_set_value( global_index, -1.0) ;
//                    //if(i == j +3 || i == j -3) test_matrix2.global_set_value( global_index, 0.3);
//                    
//                }
//            } 
//        }
//        ptr_comm->barrier();
//    }
//    std::cout << test_matrix2 <<std::endl;
//    //std::cout << "2-" << ptr_comm->get_rank() <<std::endl;
//    Contiguous1DMap map2 (std::array<int,1>({N}),  ptr_comm->get_rank(), ptr_comm->get_world_size() );
//
////    SE::DenseTensor<1,double,Contiguous1DMap<1>, DEVICETYPE::MPI> test_matrix2(*ptr_comm,);
////    std::cout <<  TensorOp::matmul( test_matrix2, test_matrix2 ) <<std::endl;
////
////    int myrank = comm->rank;
////    int nprocs = comm->world_size;
////    /*      
////    int nn=100000;
////    double step = 0.00001;
////
////    int myinit = myrank*(nn/nprocs);
////    int myfin =  (myrank+1)*(nn/nprocs)-1;
////    if(myfin > nn) myfin = nn;
////    if(myrank == 0) { std::cout << "allreduce test" << std::endl;}
////    comm->barrier();
////    std::cout << "myrank : " << myrank << ", myinit : " << myinit << ", myfin : " << myfin << std::endl;
////    for(int i = myinit ; i<=myfin ; i++){
////        x = ((double)i+0.5)*step;
////        sum = sum + 4.0/(1.0+x*x);
////    }
////    double tsum = 0.0;
////    comm->barrier();
////    comm->allreduce<double>(&sum,1,&tsum,SEop::SUM);
////    //MPI_Allreduce(&sum, &tsum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
////    std::cout.precision(10);
////    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
////    sum = tsum = 0;
////    
////    comm->barrier();
////    for(int i = myrank ; i<nn ;i=i+nprocs){
////        x = ((double)i+0.5)*step;
////        sum = sum + 4.0/(1.0+x*x);
////    }
////    comm->allreduce(&sum,1,&tsum,SEop::SUM);
////    std::cout.precision(10);
////    std::cout << "myrank : " << myrank << ", sum = " << sum << ", tsum*step = " << tsum*step << std::endl;
////
////    if(myrank == 0) { std::cout << "alltoall test" << std::endl;}
////    double* irecv = (double *)malloc(sizeof(double)*100);
////    double* test_array = (double *)malloc(sizeof(double)*100);
////    for(int i=0;i<100;i++){
////        test_array[i] = (double)i*(myrank+1);
////    }
////    comm->alltoall(test_array,100/nprocs,irecv,100/nprocs);    
////    if(myrank == 1) {
////        for(int i=0;i<100;i++){
////            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
////        }
////    }
////    comm->barrier();
////    if(myrank == 1) { std::cout << "allgather test" << std::endl;}
////
////    comm->allgather(test_array,100/nprocs,irecv,100/nprocs);
////    if(myrank == 1) {
////        for(int i=0;i<100;i++){
////            std::cout << i << " : " << test_array[i] << " " <<  irecv[i] << std::endl;
////        }
////    }
////    comm->barrier();
////    */
////    /*
////    std::cout << "ContiguousMap test" << std::endl;
////    
////    std::array<int,3> shape3 = {8,7,17}; 
////    
////    ContiguousMap<3> cont_map = ContiguousMap<3>(shape3);
////    //nproc = 3
////    std::array<int,3> test_index1 = {1,3,14}; // 1+ 3*8 + 14*8*7 = 809
////    int test_index1_ = 809;
////    if(comm->rank == 0){
////        int slice_dimension = comm->rank;
////        //array G -> array L
////        std::array<int,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        int local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
////        //int G -> array L
////        std::array<int,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        int local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*2 + 14*2*7 = 203
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (1 3 14 ) = 203std::endl; // rank 0 : (1 3 14 ) = 203
////        
////        //array L -> array G
////        std::array<int, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        std::array<int, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        int restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        int restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    if(comm->rank == 1){
////        int slice_dimension = comm->rank;
////        //array G -> array L
////        std::array<int,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        int local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
////        //int G -> array L
////        std::array<int,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        int local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 1*8 + 14*8*2 = 233
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (1 1 14 ) = 233
////        
////        //array L -> array G
////        std::array<int, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        std::array<int, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        int restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        int restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    if(comm->rank == 2){
////        int slice_dimension = comm->rank;
////        //array G -> array L
////        std::array<int,3> sliced = cont_map.get_local_array_index(test_index1, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        int local = cont_map.get_local_index(test_index1, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
////        //int G -> array L
////        std::array<int,3> sliced_ = cont_map.get_local_array_index(test_index1_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        int local_ = cont_map.get_local_index(test_index1_, slice_dimension, comm->rank, comm->world_size); // 1+ 3*8 + 4*8*7 = 249
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (1 3 4 ) = 249
////        
////        //array L -> array G
////        std::array<int, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        std::array<int, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        int restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        int restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    comm->barrier();
////
////    std::array<int,3> test_index2 = {7,6,16}; // 7+ 6*8 + 16*8*7 = 951
////    int test_index2_ = 951;
////    if(comm->rank == 2){
////        ////////////////////////////////////
////        int slice_dimension = 0;
////        //array G -> array L
////        std::array<int,3> sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        int local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
////        //int G -> array L
////        std::array<int,3> sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        int local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 3+ 6*4 + 16*4*7 = 475
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 0 : (3 6 16 ) = 475
////        //array L -> array G
////        std::array<int, 3> restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        std::array<int, 3> restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        int restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        int restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////        ////////////////////////////////////
////        slice_dimension = 1;
////        //array G -> array L
////        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
////        //int G -> array L
////        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 2*8 + 16*8*3 = 407
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 1 : (7 2 16 ) = 407
////        //array L -> array G
////        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////        ////////////////////////////////////
////        slice_dimension = 2;
////        //array G -> array L
////        sliced = cont_map.get_local_array_index(test_index2, slice_dimension, comm->rank, comm->world_size);
////        //array G -> int L
////        local = cont_map.get_local_index(test_index2, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
////        //int G -> array L
////        sliced_ = cont_map.get_local_array_index(test_index2_, slice_dimension, comm->rank, comm->world_size);
////        //int G -> int L
////        local_ = cont_map.get_local_index(test_index2_, slice_dimension, comm->rank, comm->world_size); // 7+ 6*8 + 6*8*7 = 391
////        std::cout << "rank " << comm->rank << " : " << sliced << " = " << local << ", " << sliced_ << " = " << local_ << std::endl; // rank 2 : (7 6 6 ) = 391
////        //array L -> array G
////        restored1 = cont_map.get_global_array_index(sliced, slice_dimension, comm->rank, comm->world_size);
////        //int L -> array G
////        restored2 = cont_map.get_global_array_index(local, slice_dimension, comm->rank, comm->world_size);
////        //array L -> int G
////        restored1_ = cont_map.get_global_index(sliced_, slice_dimension, comm->rank, comm->world_size);
////        //int L -> int G
////        restored2_ = cont_map.get_global_index(local_, slice_dimension, comm->rank, comm->world_size);
////        std::cout << "rank " << comm->rank << " : " << restored1 << ", " << restored2 << ", " << restored1_ << ", " << restored2_ << std::endl; 
////    }
////    comm->barrier();
////    */
////    /*
////    std::array<int, 2> test_shape = {3,3};
////    std::vector<double> test_data = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
////    ContiguousMap<2> new_map = ContiguousMap<2>(test_shape);
////    SE::DenseTensor<double, 2, Comm<DEVICETYPE::MPI>, ContiguousMap<2> > test_matrix
////                = SE::DenseTensor<double, 2, Comm<DEVICETYPE::MPI>, ContiguousMap<2> > (test_shape, &test_data[0]);
////    auto out = test_matrix.decompose("EVD");
////    if (comm->rank == 0){
////        print_eigenvalues( "Eigenvalues", out.get()->num_eig, out.get()->real_eigvals.get(), out.get()->imag_eigvals.get());
////    }
////    comm->barrier();
////    */
////    std::cout << "Sparse Matrix test" << std::endl;
////
////    int N = 30;
////    std::array<int, 2> test_shape2 = {N,N};
////    ContiguousMap<2> new_map2(test_shape2,  comm->get_rank() , comm->get_world_size()  );
////    SE::SparseTensor<2, double, Contiguous1DMap<2> > test_sparse(comm.get(), new_map2, test_shape2);
////
////    
////    std::cout << "Matrix construction" << std::endl;
////    int chunk_size = test_sparse.shape[0] / test_sparse.comm->world_size;
////    int* local_matrix_size = new_map2.get_partition_size_array();
////
////    for(int i=0;i<N;i++){
////    //for(int loc_i=0; loc_i<local_matrix_size[myrank]; loc_i++){
////        for(int j=0;j<N;j++){
////            std::array<int,2> index = {i,j};
////            if(new_map2.get_my_rank_from_global_index(index)==myrank){
////                if(i == j)                   test_sparse.insert_value(index, 2.0*((double)i+1.0-(double)N));
////                //if(i == j +1 || i == j -1)   test_sparse.insert_value(index, 3.0);
////                if(i == j +2 || i == j -2)   test_sparse.insert_value(index, -1.0);
////                if(i == j +3 || i == j -3)   test_sparse.insert_value(index, 0.3);
////                //if(i == j +4 || i == j -4)   test_sparse.insert_value(index, -0.1);
////            }
////            else{
////                continue;
////            }
////        }
////    }
////    test_sparse.complete();
////    std::cout << "matrix construction complete" << std::endl;
////    //test_sparse.print();
////    /*
////    comm->barrier();
////    if (comm->rank == 0) std::cout << "Sparsematrix Davidson" << std::endl;
////    std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();  
////    auto out3 = decompose(test_sparse, "davidson");
////    if (comm->rank == 0) print_eigenvalues( "Eigenvalues", 3, out3.get()->real_eigvals.get(), out3.get()->imag_eigvals.get());
////    comm->barrier();
////    std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
////    if (comm->rank == 0) std::cout << "BlockDavidson_sparse, calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count())/1000000.0 << "[sec]" << std::endl;
////   */
////
////
////
//
//    return 0;
//}
