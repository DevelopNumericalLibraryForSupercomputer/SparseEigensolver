#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <string>
#include <chrono>
#include <random>
#include <utility>
#include <fstream>
#include <regex>

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

std::pair<int, std::vector<double> > read_file(const std::string filename){
    std::ifstream matrix_file(filename);
    //std::vector<std::vector<int>> matrix;
    
	std::pair<int, std::vector<double> > output;
	output.first=0;

    if (!matrix_file.is_open()) {
        std::cerr << "Could not open the file " << filename << std::endl;
        return output;
    }

    std::string line;
    std::string cell;
	std::regex rgx("\\s+");
    while (std::getline(matrix_file, line)) {
        std::stringstream lineStream(line);
		std::sregex_token_iterator reg_end;
		std::sregex_token_iterator iter(line.begin(), line.end(), rgx, -1);
		for ( ; iter != reg_end; ++iter){
			if(""==iter->str()) continue;
            output.second.push_back(std::stod(iter->str()));
        }
		output.first++;
    }
    matrix_file.close();
	return output;
}

int main(int argc, char** argv){
	// predefined value
    int rank=0, nprocs=1, ictxt;
	DecomposeOption option;
	std::string  filename="Sparse_Hamiltonian_sz_+-2.000000.txt";

	// input 
	int p=1;
	if (argc>=2 ) p = std::stoi( argv[1] ); 
	
	int q=1;
	if (argc>=3 ) q = std::stoi( argv[2] );
    
	int nb = 2;
	if (argc>=4 ) nb=std::stoi(argv[4]);
	//int p=1; int q=1;
	int precond_type = 2;
	if (argc>=5 ) option.preconditioner = (PRECOND_TYPE) std::stoi(argv[5]);

	int file_number= 2;
	if (argc>=6 ) filename = "Sparse_Hamiltonian_sz_+-"+std::to_string(file_number)+".000000.txt";


	MPICommInp comm_inp({p,q});
    auto ptr_comm = comm_inp.create_comm();

    if(ptr_comm->get_rank()==0) std::cout << "========================\nDense matrix davidson test" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 1 read matrix
    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();  
	auto output = read_file(filename);
	const int N = output.first;
	assert ( output.second.size() ==N*N);
    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "reading matrix takes" << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count())/1000000.0 << "[sec]" << std::endl;

	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 2 construct matrices
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
	// num_eig variable is not used 
    const int num_eig = 3;

	if (argc==1 and ptr_comm->get_rank()==0){
		std::cout << "p: number of rows in processor grid\n" 
                  << "q: number of columns in processor grid\n" 
                  << "nb: number of block size\n" 
                  << "precond_type: 1 (diagonal) 2 (ISI2)\n" 
                  << "file_number: 0,1,2 "
                  << std::endl; 
		return 0;
	}

	if(ptr_comm->get_rank()==0) std::cout << "Dimension: " <<    N<<std::endl;

    // Set the seed for random number generation
    std::mt19937 rng(12345);  // Fixed seed number for reproducibility
    std::uniform_real_distribution<double> dist(-1, 1);  // Define the range of random values


	BlockCyclingMapInp<2> map_inp({N,N}, ptr_comm->get_rank(), ptr_comm->get_world_size(), {nb, nb}, comm_inp.nprow );
	DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> test_matrix(comm_inp.create_comm(), map_inp.create_map() );
	memcpy<double, DEVICETYPE::MPI> ( test_matrix.data.get(), output.second.data(), output.second.size() );
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

    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "constructing matrices takes" << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 3 diagonalization
    if(ptr_comm->get_rank()==0) std::cout << "========================\nDense matrix diag start" << std::endl;
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(test_matrix, guess, option);
    if(ptr_comm->get_rank()==0) print_eigenvalues( "Eigenvalues", num_eig, out1.get()->real_eigvals.data(), out1.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "block davidson calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    delete guess;
  return 0;
}

