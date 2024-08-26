#include <iostream>
#include <math.h>
#include <cassert>
#include "DenseTensor.hpp"
#include "device/mpi/TensorOp.hpp"
#include "device/mpi/LinearOp.hpp"
#include "decomposition/Preconditioner.hpp"
#include "decomposition/DecomposeOption.hpp"

//#include "device/mkl/TensorOp.hpp"

using namespace SE;


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
void fill_matrix (DenseTensor<2,DATATYPE,mtype,device>& matrix, const bool print = false, TRANSTYPE trans = TRANSTYPE::N){
	const int n = matrix.ptr_map->get_global_shape(0);
	const int m = matrix.ptr_map->get_global_shape(1);
	//assert (m == n);

	const DATATYPE  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;

	if (print and (matrix.ptr_comm->get_rank()==0)  ) {
		printf("fill_matrix==================================================================\n");
	}

    for ( int j=0; j<n; j++ ){
        for ( int i=0; i<m; i++ ){
			std::array<int,2> global_array_index;
			if (trans==TRANSTYPE::N){
				global_array_index = {j,i};
			}
			else{
				global_array_index = {i,j};
			}
			auto local_index = matrix.ptr_map->unpack_local_array_index(matrix.ptr_map->global_to_local(global_array_index));
			if(local_index<0) continue;
			if(local_index>=matrix.ptr_map->get_num_local_elements()) continue;

            if ( j < n-1 ){
                if ( i <= j ){
                    matrix.global_set_value( global_array_index,  one / sqrt( ( DATATYPE )( (j+1)*(j+2) ) ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << one / sqrt( ( DATATYPE )( (j+1)*(j+2) ) ) <<" ";
                } else if ( i == j+1 ) {
                    matrix.global_set_value( global_array_index, -one / sqrt( one + one/( DATATYPE )(j+1) ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << -one / sqrt( one + one/( DATATYPE )(j+1) )  <<" ";
                } else {
                    matrix.global_set_value( global_array_index, zero);
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout << zero  << " ";
                }
            } else {
                matrix.global_set_value( global_array_index, one / sqrt( ( DATATYPE )n ) );
					if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout <<  one / sqrt( ( DATATYPE )n )  << " ";
            }

        }
		if(print and (matrix.ptr_comm->get_rank()==0) ) std::cout  <<std::endl;
	}
	return;
}



#define DATATYPE double
#define mtype    MTYPE::BlockCycling
#define device   DEVICETYPE::MPI

int main(int argc, char* argv[]){
    int rank=0, nprocs=1, ictxt;
	DecomposeOption option;
	// problem define 

//	Comm<DEVICETYPE::MKL> comm(rank,nprocs);
//	Contiguous1DMap<2> map(global_shape, rank, nprocs);
//	DenseTensor<2, DATATYPE, SE::Contiguous1DMap<2>, DEVICETYPE::MKL > A(comm, map);
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
	int precond_type = 2;
	if (argc>=6 ) option.preconditioner = (PRECOND_TYPE) std::stoi(argv[5]);

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

	DenseTensor<2, DATATYPE, mtype, device > A(ptr_mpi_comm, ptr_block_map);
	//auto scope = 'A';
	//blacs_barrier(&ictxt,&scope);
	fill_matrix(A, true);
	DenseTensor<2, DATATYPE, mtype, device > B(ptr_mpi_comm, ptr_block_map);
	fill_matrix(B, true, TRANSTYPE::T);
	TensorOp::add_(A, B, 1.0);

	BlockCyclingMapInp<2> block_map_inp2({n,option.num_eigenvalues},
                                        ptr_mpi_comm->get_rank(), 
	                                    ptr_mpi_comm->get_world_size(), 
								        block_size, nprow);
	
	ptr_block_map = block_map_inp2.create_map();

	DenseTensor<2, DATATYPE, mtype, device > guess(ptr_mpi_comm, ptr_block_map);
	//blacs_barrier(&ictxt,&scope);
	fill_matrix(guess, true);

	DenseTensorOperations operation(A);
	auto preconditioner  = get_preconditioner<DATATYPE, mtype, device> (&operation, option);
	DATATYPE* sub_eigval = new DATATYPE[option.num_eigenvalues];
	DATATYPE* norm = new DATATYPE[option.num_eigenvalues];

    auto sub_matrix = TensorOp::matmul( *TensorOp::conjugate(guess), TensorOp::matmul(A, guess), TRANSTYPE::T );	
	auto sub_eigvec = TensorOp::diagonalize<DATATYPE,  mtype, device> ( sub_matrix, sub_eigval );
	//std::cout << std::setprecision(6) << sub_eigvec <<std::endl;
	//std::cout << std::setprecision(6) << sub_eigval[0] << ", " <<sub_eigval[1] <<", " <<sub_eigval[2] <<std::endl;
	auto eigvec = TensorOp::matmul(guess,sub_eigvec);
	auto residual = TensorOp::add(TensorOp::matmul(A, eigvec), *TensorOp::scale_vectors(eigvec, sub_eigval), -1.0); 

	auto output = preconditioner->call(*residual, sub_eigval );
	
	for (int i=0; i<output->ptr_map->get_local_shape(0); i++){
		for (int j =0; j<output->ptr_map->get_local_shape(1); j++){
			std::array<int,2> arr_idx = {i,j};
			auto global_arr_idx = output->ptr_map->local_to_global(arr_idx);
			printf("%d %d %f\n",global_arr_idx[0], global_arr_idx[1],  output->data[output->ptr_map->unpack_local_array_index(arr_idx) ]);
		}
	}
	//blacs_exit(&ictxt);
	delete[] sub_eigval;
	delete[] norm;
	return 0;
}
