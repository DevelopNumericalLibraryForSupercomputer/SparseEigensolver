#include <iostream>
#include <math.h>
#include <cassert>
#include "DenseTensor.hpp"
#include "device/mpi/TensorOp.hpp"
#include "device/mpi/LinearOp.hpp"
//#include "device/mkl/TensorOp.hpp"

using namespace SE;

template<MTYPE mtype, DEVICETYPE device>
void fill_matrix (DenseTensor<2,double,mtype,device>& matrix, const bool print = false){
	const int n = matrix.ptr_map->get_global_shape(0);
	const int m = matrix.ptr_map->get_global_shape(1);
	assert (m == n);

	const double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;

	if (print and (matrix.ptr_comm->get_rank()==0)  ) {
		printf("fill_matrix==================================================================\n");
	}

    for ( int j=0; j<n; j++ ){
        for ( int i=0; i<m; i++ ){
            if ( j < n-1 ){
                if ( i <= j ){
                    matrix.global_set_value( i+n*j ,  one / sqrt( ( double )( (j+1)*(j+2) ) ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << one / sqrt( ( double )( (j+1)*(j+2) ) ) <<" ";
                } else if ( i == j+1 ) {
                    matrix.global_set_value( i+n*j , -one / sqrt( one + one/( double )(j+1) ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << -one / sqrt( one + one/( double )(j+1) )  <<" ";
                } else {
                    matrix.global_set_value( i+n*j , zero);
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << zero  << " ";
                }
            } else {
                matrix.global_set_value( i+n*(n-1) , one / sqrt( ( double )n ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout <<  one / sqrt( ( double )n )  << " ";
            }

        }
		if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout  <<std::endl;
	}
	return;
}

int main(int argc, char* argv[]){
    int rank=0, nprocs=1, ictxt;
	// problem define 

//	Comm<DEVICETYPE::MKL> comm(rank,nprocs);
//	Contiguous1DMap<2> map(global_shape, rank, nprocs);
//	DenseTensor<2, double, SE::Contiguous1DMap<2>, DEVICETYPE::MKL > A(comm, map);
//	fill_matrix(A);

	// input 
	int p=1;
	if (argc>=2 ) p = std::stoi( argv[1] ); 
	
	int q=1;
	if (argc>=3 ) q = std::stoi( argv[2] );
    
	int n = 6000;
	if (argc>=4 ) n=std::stoi(argv[3]);

	int nb = 2;
	if (argc>=5 ) nb=std::stoi(argv[4]);
	//int p=1; int q=1;

	// num_eig variable is not used 
	std::array<int, 2> global_shape = {n,n};
	std::array<int, 2> block_size = {nb,nb};
	std::array<int, 2> nprow = {p,q};  // only np==p*q works 

	const int i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
	int info;
	int descA[9];
	int descB[9];
	int descC[9];
	//blacs_pinfo( &rank, &nprocs );
    //blacs_get( &i_negone, &i_zero, &ictxt );
    //blacs_gridinit( &ictxt, "C", &nprow[0], &nprow[1] );

	MPICommInp mpi_comm_inp(nprow);
	auto ptr_mpi_comm = mpi_comm_inp.create_comm();
	BlockCyclingMapInp<2> block_map_inp(global_shape, 
                                        ptr_mpi_comm->get_rank(), 
	                                    ptr_mpi_comm->get_world_size(), 
								        block_size, nprow);
	
	auto ptr_block_map = block_map_inp.create_map();

	DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI > A(ptr_mpi_comm, ptr_block_map);
	auto scope = 'A';
	blacs_barrier(&ictxt,&scope);
	fill_matrix(A, true);


	double norm [ n ];
	SE::TensorOp::get_norm_of_vectors(A, norm, n);
	for (int i=0; i<n; i++){
		printf("%d \t %f\n",i, norm[i])	;
	}

	for (int i=0; i<n; i++){
		norm[i] = i;
	}

	blacs_barrier(&ictxt,&scope);
	if(ptr_mpi_comm->get_rank()==0) printf("=============================================================================\n");

	for (int i=0; i<A.ptr_map->get_local_shape(0); i++){
		for (int j =0; j<A.ptr_map->get_local_shape(1); j++){
			std::array<int,2> arr_idx = {i,j};
			auto global_arr_idx = A.ptr_map->local_to_global(arr_idx);
			printf("%d %d %f\n",global_arr_idx[0], global_arr_idx[1],  A.data[A.ptr_map->unpack_local_array_index(arr_idx) ]);
		}
	}
	blacs_barrier(&ictxt,&scope);
	if(ptr_mpi_comm->get_rank()==0) printf("=============================================================================\n");
	blacs_barrier(&ictxt,&scope);
	//SE::TensorOp::scale_vectors(A, norm);
	SE::TensorOp::orthonormalize(A, "nothing");

	for (int i=0; i<A.ptr_map->get_local_shape(0); i++){
		for (int j =0; j<A.ptr_map->get_local_shape(1); j++){
			std::array<int,2> arr_idx = {i,j};
			auto global_arr_idx = A.ptr_map->local_to_global(arr_idx);
			printf("%d %d %f\n",global_arr_idx[0], global_arr_idx[1],  A.data[A.ptr_map->unpack_local_array_index(arr_idx) ]);
		}
	}
	//blacs_exit(&ictxt);
	return 0;
}
