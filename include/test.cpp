#include "mkl_blacs.h"
#include "mkl_pblas.h"
#include "mkl_scalapack.h"
#include "BlockCyclingMap.hpp"
#include "DenseTensor.hpp"
#include <cassert>
#include <math.h>
#include "device/mpi/TensorOp.hpp"

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

int main(int argc, char* argv[]){
    int rank, world_size, ictxt;
	const double  zero = 0.0E+0, one = 1.0E+0, two = 2.0E+0, negone = -1.0E+0;
	const int i_zero = 0, i_one = 1, i_four = 4, i_negone = -1;
	int info;
	int descA[9];
	int descB[9];
	int descC[9];

	// problem define 
	const int n =51;
	const int nb = 3;
	int p=2; int q=2;

	std::array<int, 2> global_shape = {n,n};
	std::array<int, 2> block_size = {nb,nb};
	std::array<int, 2> nprow = {p,q};  // only np 4 works 

    blacs_pinfo( &rank, &world_size );
    blacs_get( &i_negone, &i_zero, &ictxt );
    blacs_gridinit( &ictxt, "C", &nprow[0], &nprow[1] );
    //blacs_gridinfo( &ictxt, &nprow[0], &npcol[1], &myrow, &mycol );

//    if ( rank == 0 ) {
//        printf( " %d %d\n", rank, world_size );
//	}

    SE::BlockCyclingMap<2> map(  global_shape, rank, world_size, block_size, nprow);
	for (int i=0; i<global_shape[0]; i++){
		for (int j=0; j<global_shape[1]; j++){
//			std::array<int,2> index1 = {i,j};
//			auto index2=  map.global_to_local( index1 );
//			auto idx1 = map.unpack_global_array_index( index1);
//			auto idx2 = map.unpack_local_array_index( index2);
//
//			if(rank==3) printf("%d: %d %d %d %d \n",rank, index1[0], index1[1], index2[0], index2[1]);
//			if(rank==3) printf("%d: %d %d\n",rank,  idx1, idx2);
			int idx = i+n*j;
			auto idx_ = map.pack_global_index(idx);
			auto local_idx = map.global_to_local(idx);
			//if(rank==0) printf("%d: %d %d\n", idx, idx_[0], idx_[1] );
		}
	}
	SE::Comm<SE::DEVICETYPE::MPI> comm(rank, world_size);
	SE::DenseTensor<2, double, SE::BlockCyclingMap<2>, SE::DEVICETYPE::MPI > A(comm, map);
	SE::DenseTensor<2, double, SE::BlockCyclingMap<2>, SE::DEVICETYPE::MPI > B(comm, map);
	SE::DenseTensor<2, double, SE::BlockCyclingMap<2>, SE::DEVICETYPE::MPI > C(comm, map);

	
    for (int i=0; i<n; i++ ){
        for ( int j=0; j<n; j++ ){
            B.global_set_value( i+n*j ,  one*rand()/RAND_MAX);
        }
        B.global_insert_value( i+n*i ,two);
    }
	double normA = 0.0;
    for ( int j=0; j<n; j++ ){
        for ( int i=0; i<n; i++ ){
            if ( j < n-1 ){
                if ( i <= j ){
                    A.global_set_value( i+n*j ,  one / sqrt( ( double )( (j+1)*(j+2) ) ) );
					normA+= pow( one / sqrt( ( double )( (j+1)*(j+2) ) ) , 2.0);
                } else if ( i == j+1 ) {
                    A.global_set_value( i+n*j , -one / sqrt( one + one/( double )(j+1) ) );
					normA+= pow( -one / sqrt( one + one/( double )(j+1) )  , 2.0);
                } else {
                    A.global_set_value( i+n*j , zero);
                }
            } else {
                A.global_set_value( i+n*(n-1) , one / sqrt( ( double )n ) );
				normA+= pow( one / sqrt( ( double )n ) , 2.0 );
            }
        }
    }

	double* work = new double[A.map.get_num_local_elements()];
	int myrow = A.map.get_my_array_rank()[0];
	int mycol = A.map.get_my_array_rank()[1];

    int lld = MAX( A.map.get_local_shape()[0], 1 );

//	std::cout << rank << "\t"<<  myrow  <<"\t" <<mycol << "\t" <<lld<<"\t" <<"\t"<<numroc( &n, &nb, &myrow, &i_zero, &nprow[0] ) <<std::endl;
//	std::cout << rank << "\t"<<  myrow  <<"\t" <<mycol << "\t" <<lld <<"\t"<<numroc( &n, &nb, &myrow, &i_zero, &nprow[0] ) <<std::endl;
/*  Initialize descriptors for distributed arrays */
    descinit( descA, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
	assert (info==0);
    descinit( descB, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
	assert (info==0);
    descinit( descC, &n, &n, &nb, &nb, &i_zero, &i_zero, &ictxt, &lld, &info );
	assert (info==0);
    
	double anorm, bnorm;
	anorm = pdlange( "F", &n, &n, A.data, &i_one, &i_one, descA, work );
    bnorm = pdlange( "F", &n, &n, B.data, &i_one, &i_one, descB, work );

	auto result1 = SE::TensorOp::matmul(A, B, SE::TRANSTYPE::N, SE::TRANSTYPE::N);
	auto result2 = SE::TensorOp::matmul(A, result1, SE::TRANSTYPE::T, SE::TRANSTYPE::N);
	auto result3 = SE::TensorOp::add( B, result2, -1.0);

	// C=A@B	
    pdgemm( "N", "N", &n, &n, &n, &one, A.data, &i_one, &i_one, descA, B.data, &i_one, &i_one, descB,
             &zero, C.data, &i_one, &i_one, descC );
	// B=inv_A@C
    pdgemm( "T", "N", &n, &n, &n, &one, A.data, &i_one, &i_one, descA, C.data, &i_one, &i_one, descC,
             &negone, B.data, &i_one, &i_one, descB );

	double diffnorm1 = pdlange( "I", &n, &n, B.data, &i_one, &i_one, descB, work );
	double diffnorm2 = pdlange( "I", &n, &n, result3.data, &i_one, &i_one, descB, work );
	if( rank == 2 ){ 
		//printf("%03.11f\n", sqrt(normA) );
		printf( ".. Norms of A and B are computed ( p?lange ) ..\n" ); 
        printf( "||A|| = %03.11f\n", anorm );
        printf( "||B|| = %03.11f\n", bnorm );
		printf(" %03.11f %03.11f\n\n", diffnorm1, diffnorm2);
	}

//	std::cout <<A <<std::endl;
//	std::cout <<B <<std::endl;
//	std::cout <<C <<std::endl;
//	std::cout << C <<std::endl;
	// B = B-inv_A@C  (inv_A==A_T, because A is orthonormal)
	//delete[] work;

//	std::cout <<"A" <<std::endl;
//	std::cout  << A <<std::endl;
//
//	std::cout <<"B" <<std::endl;
//	std::cout << B <<std::endl;
///*  Print information of task */
//    printf( "=== START OF EXAMPLE ===================\n" );
//    printf( "Matrix-matrix multiplication: A*B = C\n\n" );
//    printf( "/  1/q_1 ........   1/q_n-1     1/q_n  \\ \n" );
//    printf( "|        .                             | \n" );
//    printf( "|         `.           :         :     | \n" );
//    printf( "| -1/q_1    `.         :         :     | \n" );
//    printf( "|        .    `.       :         :     |  =  A \n" );
//    printf( "|   0     `.    `                      | \n" );
//    printf( "|   : `.    `.      1/q_n-1     1/q_n  | \n" );
//    printf( "|   :   `.    `.                       | \n" );

    blacs_gridexit( &ictxt );
    blacs_exit( &i_zero );
	return 0;
}
