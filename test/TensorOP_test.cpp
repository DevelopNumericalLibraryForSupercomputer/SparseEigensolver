#include <vector>
#include <array>
#include <iostream>
#include <iomanip>
#include "ContiguousMap.hpp"
#include "device/MKL/TensorOp.hpp"
#include "device/MPI/TensorOp.hpp"

using namespace SE;
int serial(int argc, char* argv[]){
    std::cout << "TensorOp test, MKL" << std::endl;
    
    auto comm = createComm<SEMkl>(argc, argv);
    //std::unique_ptr<Comm<SEMkl> > comm = createComm<SEMkl>(argc, argv);
    std::cout << "SERIAL test" << std::endl;
    std::cout << comm.get()->world_size << std::endl;
    
    std::array<size_t, 2> test_shape = {3,3};
    ContiguousMap<2>* map = new ContiguousMap(test_shape, 1);
    
    std::vector<double> data1 = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    std::vector<double> data2 = {0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0};

    auto matrix1 = new Tensor<STORETYPE::Dense, double, 2, SEMkl, ContiguousMap<2> >(comm.get(), map, test_shape, &data1[0]);
    auto matrix11 = matrix1->clone();
    auto matrix2 = new Tensor<STORETYPE::Dense, double, 2, SEMkl, ContiguousMap<2> >(comm.get(), map, test_shape, &data2[0]);
    auto matrix3 = matmul(matrix1, matrix2);

    
    matrix1->print("mat1");
    matrix11->print("mat11");
    matrix2->print("mat2");
    matrix3->print("mat3");

    size_t N = 10;
    std::array<size_t, 2> test_shape2 = {N,N};
    ContiguousMap<2>* big_map = new ContiguousMap(test_shape2, 1);
    auto dense = new Tensor<STORETYPE::Dense, double, 2, SEMkl, ContiguousMap<2> >(comm.get(), big_map, test_shape2);
    auto sparse = new Tensor<STORETYPE::COO, double, 2, SEMkl, ContiguousMap<2> >(comm.get(), big_map, test_shape2, N*3);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(i == j){
                sparse->insert_value(index, 2.0*((double)i+1.0-(double)N) );
                dense->insert_value(index, 2.0*((double)i+1.0-(double)N) );
            }
            if(i == j +2 || i == j -2){
                sparse->insert_value(index, -1.0);
                dense->insert_value(index, -1.0);
            }
            if(i == j +3 || i == j -3){
                sparse->insert_value(index, 0.3);
                dense->insert_value(index, 0.3); 
            }
        }
    }
    sparse->complete();
    std::cout << "matrix construction complete" << std::endl;
    sparse->print("sparse");
    dense->print("dense");

    std::array<size_t, 1> shape_vec = {N};
    ContiguousMap<1>* vec_map = new ContiguousMap(shape_vec, 1);
    double* vec_entity = malloc<double, SEMkl>(N);
    memset<double, SEMkl>(vec_entity, 0, N);
    vec_entity[3] = 1;
    vec_entity[1] = 1;
    auto onedvec = new Tensor<STORETYPE::Dense, double, 1, SEMkl, ContiguousMap<1> >(comm.get(), vec_map, shape_vec, vec_entity);
    auto coo_spmv_prod = spmv(sparse, onedvec, SE_transpose::NoTrans);
    auto dense_spmv_prod = spmv(dense, onedvec, SE_transpose::NoTrans);
    onedvec->print("onedvec");
    coo_spmv_prod->print("coo_spmv_prod");
    dense_spmv_prod->print("dense_spmv_prod");

    std::array<size_t, 2> shape_three_vecs = {N, 3};
    ContiguousMap<2>* three_vecs_map = new ContiguousMap(shape_three_vecs, 1);
    double* vecs_entity = malloc<double, SEMkl>(N*3);
    memset<double, SEMkl>(vecs_entity, 0, N*3);
    vecs_entity[0] = 1;
    vecs_entity[N+1] = 1;
    vecs_entity[2*N+2] = 1;
    auto threevecs = new Tensor<STORETYPE::Dense, double, 2, SEMkl, ContiguousMap<2> >(comm.get(), three_vecs_map, shape_three_vecs, vecs_entity);
    threevecs->print("threevecs");
    auto dense_spmv_prod2 = spmv(dense, threevecs, SE_transpose::NoTrans);
    dense_spmv_prod2->print("dense_spmv_prod2");
    auto coo_spmv_prod2 = spmv(sparse, threevecs, SE_transpose::NoTrans);
    coo_spmv_prod2->print("coo_spmv_prod2");
    return 0;
}

int MPI_noslice(int argc, char* argv[]){
    std::cout << "TensorOp test, MPI" << std::endl;
    
    std::unique_ptr<Comm<SEMpi> > comm = createComm<SEMpi>(argc, argv);
    std::cout << "MPI test" << std::endl;
    std::cout << "myrank = " << comm.get()->rank << ", world_size : " << comm.get()->world_size << std::endl;

    std::cout << "============== NO SLICE : SERIAL CALCUALATION! ==============" << std::endl;
    std::array<size_t, 2> test_shape = {3,3};
    ContiguousMap<2>* map = new ContiguousMap(test_shape, comm.get()->world_size);
    
    std::vector<double> data1 = {1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0};
    std::vector<double> data2 = {0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0};

    auto matrix1 = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), map, test_shape, &data1[0]);
    auto matrix11 = matrix1->clone();
    auto matrix2 = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), map, test_shape, &data2[0]);
    auto matrix3 = matmul(matrix1, matrix2);
    comm.get()->barrier();
    matrix1->print("mat1");
    matrix11->print("mat11");
    matrix2->print("mat2");
    matrix3->print("mat3");
    comm.get()->barrier();
    size_t N = 10;
    std::array<size_t, 2> test_shape2 = {N,N};
    ContiguousMap<2>* big_map = new ContiguousMap(test_shape2, comm.get()->world_size);
    auto dense = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), big_map, test_shape2);
    auto sparse = new Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), big_map, test_shape2, N*3);
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(i == j){
                sparse->insert_value(index, 2.0*((double)i+1.0-(double)N) );
                dense->insert_value(index, 2.0*((double)i+1.0-(double)N) );
            }
            if(i == j +2 || i == j -2){
                sparse->insert_value(index, -1.0);
                dense->insert_value(index, -1.0);
            }
            if(i == j +3 || i == j -3){
                sparse->insert_value(index, 0.3);
                dense->insert_value(index, 0.3); 
            }
        }
    }
    sparse->complete();
    std::cout << "matrix construction complete" << std::endl;
    comm.get()->barrier();
    sparse->print("sparse");
    dense->print("dense");
    comm.get()->barrier();
    std::array<size_t, 1> shape_vec = {N};
    ContiguousMap<1>* vec_map = new ContiguousMap(shape_vec, comm.get()->world_size);
    double* vec_entity = malloc<double, SEMpi>(N);
    memset<double, SEMpi>(vec_entity, 0, N);
    vec_entity[3] = 1;
    vec_entity[1] = 1;
    auto onedvec = new Tensor<STORETYPE::Dense, double, 1, SEMpi, ContiguousMap<1> >(comm.get(), vec_map, shape_vec, vec_entity);
    auto coo_spmv_prod = spmv(sparse, onedvec, SE_transpose::NoTrans);
    auto dense_spmv_prod = spmv(dense, onedvec, SE_transpose::NoTrans);
    comm.get()->barrier();
    onedvec->print("onedvec");
    coo_spmv_prod->print("coo_spmv_prod");
    dense_spmv_prod->print("dense_spmv_prod");

    std::array<size_t, 2> shape_three_vecs = {N, 3};
    ContiguousMap<2>* three_vecs_map = new ContiguousMap(shape_three_vecs, 1);
    double* vecs_entity = malloc<double, SEMpi>(N*3);
    memset<double, SEMpi>(vecs_entity, 0, N*3);
    vecs_entity[0] = 1;
    vecs_entity[N+1] = 1;
    vecs_entity[2*N+2] = 1;
    auto threevecs = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), three_vecs_map, shape_three_vecs, vecs_entity);
    comm.get()->barrier();
    threevecs->print("threevecs");
    auto dense_spmv_prod2 = spmv(dense, threevecs, SE_transpose::NoTrans);
    dense_spmv_prod2->print("dense_spmv_prod2");
    auto coo_spmv_prod2 = spmv(sparse, threevecs, SE_transpose::NoTrans);
    coo_spmv_prod2->print("coo_spmv_prod2");
    free<double, SEMpi>(vec_entity);
    free<double, SEMpi>(vecs_entity);
    return 0;
}

int MPI_colslice(int argc, char* argv[]){
    std::cout << "TensorOp test, MPI" << std::endl;
    
    std::unique_ptr<Comm<SEMpi> > comm = createComm<SEMpi>(argc, argv);
    std::cout << "MPI test" << std::endl;
    std::cout << "myrank = " << comm.get()->rank << ", world_size : " << comm.get()->world_size << std::endl;

    std::cout << "============== SLICED by column index ==============" << std::endl;
    
    size_t N = 10;
    std::array<size_t, 2> test_shape2 = {N,N};
    //std::cout << comm.get()->rank << " aaa" << std::endl;
    ContiguousMap<2>* big_map = new ContiguousMap(test_shape2, comm.get()->world_size, 0);
    //std::cout << comm.get()->rank << " aaa" << std::endl;
    auto dense = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), big_map, test_shape2);
    //std::cout << comm.get()->rank << " aaa" << std::endl;
    auto sparse = new Tensor<STORETYPE::COO, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), big_map, test_shape2, N*3);
    //std::cout << comm.get()->rank << " aaa" << std::endl;
    comm.get()->barrier();    
    for(size_t i=0;i<N;i++){
        for(size_t j=0;j<N;j++){
            std::array<size_t,2> index = {i,j};
            if(i == j){
                sparse->insert_value(index, 2.0*((double)i+1.0-(double)N) );
                dense->insert_value(index, 2.0*((double)i+1.0-(double)N) );
            }
            if(i == j +2 || i == j -2){
                sparse->insert_value(index, -1.0);
                dense->insert_value(index, -1.0);
            }
            if(i == j +3 || i == j -3){
                sparse->insert_value(index, 0.3);
                dense->insert_value(index, 0.3); 
            }
        }
    }
    sparse->complete();
    std::cout << "matrix construction complete" << std::endl;
    comm.get()->barrier();
    sparse->print("sparse");
    comm.get()->barrier();
    dense->print("dense");
    comm.get()->barrier();

    
    comm.get()->barrier();
    std::cout << "std::====================================" << std::endl;
    std::array<size_t, 1> shape_vec = {N};
    ContiguousMap<1>* vec_map = new ContiguousMap(shape_vec, comm.get()->world_size);
    double* vec_entity = malloc<double, SEMpi>(N);
    memset<double, SEMpi>(vec_entity, 0, N);
    vec_entity[3] = 1;
    vec_entity[1] = 1;
    
    comm.get()->barrier();
    std::cout << "std::====================================" << std::endl;

    auto onedvec = new Tensor<STORETYPE::Dense, double, 1, SEMpi, ContiguousMap<1> >(comm.get(), vec_map, shape_vec, vec_entity);
    std::cout << "1dvec construction complete" << std::endl;
    comm.get()->barrier();
    onedvec->print("onedvec");
    comm.get()->barrier();
    std::cout << "std::====================================" << std::endl;
    std::cout << (int)sparse->map->is_sliced << " " << (int)onedvec->map->is_sliced << std::endl;
    auto coo_spmv_prod = spmv(sparse, onedvec, SE_transpose::NoTrans);
    
    comm.get()->barrier();
    coo_spmv_prod->print("coo_spmv_prod");
    comm.get()->barrier();
    std::cout << (int)dense->map->is_sliced << " " << (int)onedvec->map->is_sliced << std::endl;
    auto dense_spmv_prod = spmv(dense, onedvec, SE_transpose::NoTrans);
    comm.get()->barrier();
    dense_spmv_prod->print("dense_spmv_prod");

    std::array<size_t, 2> shape_three_vecs = {N, 3};
    ContiguousMap<2>* three_vecs_map = new ContiguousMap(shape_three_vecs, comm.get()->world_size);
    double* vecs_entity = malloc<double, SEMpi>(N*3);
    memset<double, SEMpi>(vecs_entity, 0, N*3);
    vecs_entity[0] = 1;
    vecs_entity[N+1] = 1;
    vecs_entity[2*N+2] = 1;
    auto threevecs = new Tensor<STORETYPE::Dense, double, 2, SEMpi, ContiguousMap<2> >(comm.get(), three_vecs_map, shape_three_vecs, vecs_entity);
    
    comm.get()->barrier();
    threevecs->print("threevecs");
    
    // not implemented
    //auto dense_spmv_prod2 = spmv(dense, threevecs, SE_transpose::NoTrans);
    //dense_spmv_prod2->print("dense_spmv_prod2");
    
    auto coo_spmv_prod2 = spmv(sparse, threevecs, SE_transpose::NoTrans);
    comm.get()->barrier();
    coo_spmv_prod2->print("coo_spmv_prod2");
    
    return 0;
}

int main(int argc, char* argv[]){
    //return serial(argc, argv);
    //return MPI_noslice(argc,argv);
    return MPI_colslice(argc, argv);
}