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

#include "decomposition/DecomposeOption.hpp"
#include "decomposition/Decompose.hpp"
#include "device/mpi/TensorOp.hpp"
#include "device/mpi/LinearOp.hpp"
#include "device/mpi/MPIComm.hpp"
#include "BlockCyclingMap.hpp"
//#include "decomposition/IterativeSolver_MPI.hpp"
#include "Utility.hpp"
#include "device/mkl/LinearOp.hpp"
using namespace SE;

void printMatrix(const std::vector<double>& matrix, const int rows, const int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << matrix[i * cols + j] << "\t";
        }
        std::cout << std::endl;
    }
}


std::pair<int, std::vector<double> > readMatrixFromTxtFile(const std::string filename){
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

std::pair<int, std::vector<double>> readMatrixFromBinaryFile(const std::string& filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Could not open the file " << filename << " for reading." << std::endl;
        return {0, {}};
    }

    int rows;
    int cols;
    // Read dimensions
    inFile.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    inFile.read(reinterpret_cast<char*>(&cols), sizeof(cols));

	assert(rows==cols);

	std::pair<int, std::vector<double> >output;
	output.first = rows;
	output.second.resize(rows*cols,0.0);

    // Read matrix data
    inFile.read(reinterpret_cast<char*>(output.second.data()), output.second.size() * sizeof(double));

    inFile.close();

    return output;
}

class Operations: public TensorOperations<double, MTYPE::BlockCycling, DEVICETYPE::MPI> {
public:
	Operations(int mat_size, double* matrix_elements, double* diag_elements):mat_size(mat_size),matrix_elements(matrix_elements), diag_elements(diag_elements){
		return;		
	};

    std::unique_ptr< DenseTensor<1, double, MTYPE::BlockCycling, DEVICETYPE::MPI> > matvec(const DenseTensor<1, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& vec) const{
		assert ( vec.ptr_map->get_nprow()[0]==1	 );
		
		const int m = this->mat_size;
	    const int	k = this->mat_size;

		const int contract_dim = 1;
	    const int remained_dim = 0;

		std::array<int, 1> output_shape = {this->mat_size};
		auto map_inp = vec.ptr_map->generate_map_inp();
		map_inp->global_shape = output_shape;

		std::unique_ptr< DenseTensor<1,double,MTYPE::BlockCycling, DEVICETYPE::MPI> > output = std::make_unique< DenseTensor<1,double,MTYPE::BlockCycling, DEVICETYPE::MPI> > ( vec.copy_comm(), map_inp->create_map() );

		gemv<double, DEVICETYPE::MKL>(ORDERTYPE::COL, TRANSTYPE::N, m, k, 1.0, this->matrix_elements, this->mat_size, vec.data.get(), 1, 0.0, output->data.get(), 1);
		return output;
	};
    std::unique_ptr<DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI> > matvec(const DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>& vec) const{
		assert ( vec.ptr_map->get_nprow()[0]==1	 );
        const int m = mat_size;
        const int k = mat_size;
        const int k2 = vec.ptr_map->get_local_shape(0);
        const int n = vec.ptr_map->get_local_shape(1);
        
        assert(k == k2);
        std::array<int, 2> output_shape = {m,vec.ptr_map->get_global_shape(1)};
		auto map_inp = vec.ptr_map->generate_map_inp();
		map_inp->global_shape = output_shape;
    
        std::unique_ptr<DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> > output =std::make_unique<DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> >( vec.copy_comm(), map_inp->create_map() );
		gemm<double, DEVICETYPE::MKL>(ORDERTYPE::COL, TRANSTYPE::N, TRANSTYPE::N, m, n, k, 1.0, this->matrix_elements, this->mat_size, vec.data.get(), vec.ptr_map->get_local_shape(0), 0.0, output->data.get(), vec.ptr_map->get_local_shape(0) );

    	return output;	
	};
    double get_diag_element(const int index)const{
		assert (index>=0 and index<this->mat_size);
		return this->diag_elements[index];	
	};
    std::array<int, 2> get_global_shape() const{
		return {this->mat_size, this->mat_size};	
	};
private:
	double* matrix_elements;
	double* diag_elements;
	int mat_size;
};




int main(int argc, char** argv){
	DecomposeOption option;
    option.tolerance = 1e-5;
    option.num_eigenvalues = 10;
    option.max_block = 4;
    option.max_iterations = 50;
    option.preconditioner_max_iterations = 3;

	// predefined value
    int rank=0, nprocs=1, ictxt;

	// input 
	int p=1;
	if (argc>=2 ) p = std::stoi( argv[1] ); 
	
	int q=1;
	if (argc>=3 ) q = std::stoi( argv[2] );
    
	int nb1 = 2;
	if (argc>=4 ) nb1=std::stoi(argv[3]);

	int nb2 = 2;
	if (argc>=5 ) nb2=std::stoi(argv[4]);
	//int p=1; int q=1;
	if (argc>=6 ) option.preconditioner = (PRECOND_TYPE) std::stoi(argv[5]);

	int file_number=2;
	std::string  filename;
	if (argc>=7 ) file_number = std::stoi(argv[6]);
	switch (file_number){
		case 0:
			filename = "large.dat";
			break;
		case 1:
			filename = "mid.dat";
			break;
		case 2:
			filename = "small.dat";
			break;
		case 3:
			filename = "tiny.dat";
			break;
		
	}
	MPICommInp comm_inp(std::array<int,2>({p,q}));
    auto ptr_comm = comm_inp.create_comm();
	std::array<int,2> nprow = {p,q};
    //auto ptr_comm =std::make_unique< Comm<DEVICETYPE::MPI> >( 0, p*q, nprow );
	if (argc==1){
		if(ptr_comm->get_rank()==0){
    		std::cout << "p: number of rows in processor grid\n" 
                      << "q: number of columns in processor grid\n" 
                      << "nb1: number of block size(row)\n" 
                      << "nb2: number of block size(col)\n" 
                      << "precond_type: 0 (no) 1 (Diagonal) 2 (ISI2)\n" 
                      << "file_number: 0,1,2 "
                      << std::endl; 
		}
		return 0;
	}

    if(ptr_comm->get_rank()==0){
		std::cout << "=========Dense matrix davidson test(" 
				  << p <<"," <<q<<"," <<nb1 <<"," << nb2<<"," 
				  << (int) option.preconditioner <<"," << file_number << ")" << std::endl;
	}

//	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 1 read matrix
//    std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();  
//	auto output = readMatrixFromTxtFile(filename);
//	const int N = output.first;
//	assert ( output.second.size() ==N*N);
//    std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
//    if(ptr_comm->get_rank()==0) std::cout << "reading matrix takes" << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count())/1000000.0 << "[sec]" << std::endl;
    std::chrono::steady_clock::time_point begin0 = std::chrono::steady_clock::now();  
	auto output = readMatrixFromBinaryFile(filename);
	const int N = output.first;
	assert ( output.second.size() ==N*N);
	//printMatrix(output.second, N, N);
    std::chrono::steady_clock::time_point end0 = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "reading matrix takes" << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end0 - begin0).count())/1000000.0 << "[sec]" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 2 construct matrices
    std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();  
	
	if(ptr_comm->get_rank()==0) std::cout << "Dimension: " <<    N<<std::endl;

    // Set the seed for random number generation
    std::mt19937 rng(12345);  // Fixed seed number for reproducibility
    std::uniform_real_distribution<double> dist(-1, 1);  // Define the range of random values

	auto map_inp = std::make_unique<BlockCyclingMapInp<2> > (std::array<int,2>({N,N}), ptr_comm->get_rank(), ptr_comm->get_world_size(), std::array<int,2>({nb1, nb2}), comm_inp.nprow );
	auto ptr_map =  map_inp->create_map();
	double* diag_elements = malloc<double,DEVICETYPE::MPI>(N);
	#pragma omp parallel for
	for (int i =0; i<N; i++){
		diag_elements[i] = output.second[i*N+i];
	}
	Operations op(N, output.second.data(), diag_elements);
//	DenseTensor<2,double,MTYPE::BlockCycling, DEVICETYPE::MPI> matrix(comm_inp.create_comm(), map_inp.create_map() );
//	#pragma omp parallel for
//	for (int i =0; i<matrix.ptr_map->get_num_local_elements(); i++){
//		const auto global_array_index = matrix.ptr_map->pack_global_index(matrix.ptr_map->local_to_global(i));	
//		matrix.global_set_value(global_array_index, output.second[N*global_array_index[1]+global_array_index[0]]);
//	}
//
	//memcpy<double, DEVICETYPE::MPI> ( op.data.get(), output.second.data(), output.second.size() );
	map_inp->global_shape = {N,option.num_eigenvalues};
    auto guess = std::make_unique < DenseTensor<2, double, MTYPE::BlockCycling, DEVICETYPE::MPI>> (ptr_comm, map_inp->create_map());

	//#pragma omp parallel for
	for (int i =0; i<guess->ptr_map->get_num_local_elements(); i++){
		//auto global_array_index = guess->ptr_map->pack_global_index(guess->ptr_map->local_to_global(i));	
		//guess->global_set_value(global_array_index, dist(rng));
		guess->local_set_value(i, dist(rng));
	}
//    // guess : unit vector
//    for(int i=0;i<num_eig;i++){
//		for (int j=0; j<N; j++){
//	        std::array<int, 2> tmp_index = {j,i};
//			//if (i==j)	guess->global_set_value(tmp_index, 1.0);
//			//else	guess->global_set_value(tmp_index, 1e-3);
//			guess->global_set_value(tmp_index, dist(rng));
//		}
//    }

    std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "constructing matrices takes" << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count())/1000000.0 << "[sec]" << std::endl;
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////  Part 3 diagonalization
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();  
    auto out1 = decompose(&op, guess.get(), option);
    //delete guess;
    if(ptr_comm->get_rank()==0) print_eigenvalues( "Eigenvalues", option.num_eigenvalues, out1.get()->real_eigvals.data(), out1.get()->imag_eigvals.data());
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    if(ptr_comm->get_rank()==0) std::cout << "block davidson calculation time of " << N << " by " << N << " matrix= " << ((double)std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count())/1000000.0 << "[sec]" << std::endl;
    
	if(ptr_comm->get_rank()==0) TensorOp::ElapsedTime::print();
	free<DEVICETYPE::MPI> (diag_elements);
    return 0;
}

