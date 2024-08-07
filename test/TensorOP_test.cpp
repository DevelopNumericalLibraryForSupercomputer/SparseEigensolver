#include <iostream>
#include <vector>
#include "DenseTensor.hpp"
//#include <mpi.h>
#include "device/mpi/TensorOp.hpp"


using namespace SE;

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void fill_densetensor(DenseTensor<dimension, DATATYPE, MAPTYPE, device>* mat, const std::vector<DATATYPE> data){
    assert(mat->map.get_num_global_elements() == data.size());
    for(int index = 0; index<data.size() ; ++index){
        if(mat->comm.get_rank() == mat->map.find_rank_from_global_index(index)){
            mat->global_insert_value(index,data[index]);
        }
    }
}


int main(int argc, char* argv[]){
    auto ptr_comm = create_comm<DEVICETYPE::MPI>(argc, argv);

    std::array<int, 2> mat1_shape = {7,2};
    std::array<int, 2> mat2_shape = {2,9};
    std::array<int, 2> mat3_shape = {9,9};
    std::array<int, 1> vec1_shape = {2};
    std::array<int, 1> vec2_shape = {9};
    std::array<int, 1> vec3_shape = {7};
    
    auto map_mat1 = Contiguous1DMap(mat1_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());
    auto map_mat2 = Contiguous1DMap(mat2_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());
    auto map_mat3 = Contiguous1DMap(mat3_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());
    auto map_vec1 = Contiguous1DMap(vec1_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());
    auto map_vec2 = Contiguous1DMap(vec2_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());
    auto map_vec3 = Contiguous1DMap(vec3_shape, ptr_comm->get_rank(), ptr_comm->get_world_size());

    std::vector<double> data_mat1 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14};
    std::vector<double> data_mat2 = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
    std::vector<double> data_mat3 = {1,2,3,4,5,6,7,8,9,
                                     0,0,0,0,0,0,1,0,1,
                                     1,2,3,4,5,6,7,8,9,
                                     1,2,3,4,5,6,7,8,9,
                                     1,2,3,4,5,6,7,8,9,
                                     0,0,1,0,0,0,0,1,0,
                                     1,2,3,4,5,6,7,8,9,
                                     1,2,3,4,5,6,7,8,9,
                                     1,2,3,4,5,6,7,8,9 };
    std::vector<double> data_vec1 = {1,2};
    std::vector<double> data_vec2 = {1,2,3,4,5,6,7,8,9};
    std::vector<double> data_vec3 = {1,2,3,4,5,6,7};

    //std::cout << "asdf rank = " << ptr_comm->get_rank() << " " << map_mat1.get_all_local_shape()[0][0] << " " << map_mat1.get_all_local_shape()[0][1] << std::endl;
    auto mat1 = DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MPI>(*ptr_comm, map_mat1);
    auto mat2 = DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MPI>(*ptr_comm, map_mat2);
    auto mat3 = DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MPI>(*ptr_comm, map_mat3);
    auto vec1 = DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI>(*ptr_comm, map_vec1);
    auto vec2 = DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI>(*ptr_comm, map_vec2);
    auto vec3 = DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MPI>(*ptr_comm, map_vec3);

    
    //std::cout << *mat1;
    fill_densetensor(&mat1,data_mat1);
    fill_densetensor(&mat2,data_mat2);
    fill_densetensor(&mat3,data_mat3);
    fill_densetensor(&vec1,data_vec1);
    fill_densetensor(&vec2,data_vec2);
    fill_densetensor(&vec3,data_vec3);
    //std::cout << *mat1;
    //std::cout << mat1 << mat2 << mat3 << vec1 << vec2 << vec3 << std::endl;
    std::cout << mat1 << mat2 << vec1 << vec2 << std::endl;

/*
mat1
    //  1.00  2.00 
    //  3.00  4.00 
    //  5.00  6.00
    //  7.00  8.00 
    //  9.00 10.00
    // 11.00 12.00
    // 13.00 14.00

vec1
    //1.00
    //2.00
    
mat2    
// 1.00  2.00  3.00  4.00  5.00  6.00  7.00  8.00  9.00
//10.00 11.00 12.00 13.00 14.00 15.00 16.00 17.00 18.00
    
vec2
// 1.00
// 2.00
// 3.00 
// 4.00  
// 5.00 
// 6.00  
// 7.00  
// 8.00 
// 9.00
*/ 

    //mat1 * vec1
    //  1.00  2.00    1.00     5.00
    //  3.00  4.00    2.00    11.00
    //  5.00  6.00            17.00
    //  7.00  8.00 x       =  23.00 
    //  9.00 10.00            29.00  
    // 11.00 12.00            35.00  
    // 13.00 14.00            41.00 

    std::cout << "\n\nmat1 X vec1\n" << TensorOp::matmul(mat1,vec1) << std::endl;

// 1.00  2.00  3.00  4.00  5.00  6.00  7.00  8.00  9.00  X  1.00 = 285.00
//10.00 11.00 12.00 13.00 14.00 15.00 16.00 17.00 18.00     2.00   690.00
//                                                          3.00  
//                                                          4.00  
//                                                          5.00  
//                                                          6.00  
//                                                          7.00  
//                                                          8.00  
//                                                          9.00  
    //std::cout << "\n\nmat2 X vec2\n" << TensorOp::matmul(mat2,vec2) << std::endl;

// 1.00 10.00 x 1.00 = 21.00  
// 2.00 11.00   2.00   24.00
// 3.00 12.00          27.00
// 4.00 13.00          30.00
// 5.00 14.00          33.00
// 6.00 15.00          36.00
// 7.00 16.00          39.00
// 8.00 17.00          42.00
// 9.00 18.00          45.00

    //std::cout << "\n\n mat2T*vec1" << TensorOp::matmul(mat2,vec1,TRANSTYPE::T) << std::endl;
/*
    std::cout << "\n\n mat1T*vec3" << TensorOp::matmul(mat1,vec3,TRANSTYPE::T) << std::endl;
    std::cout << "\n\n mat3 *vec2" << TensorOp::matmul(mat3,vec2,TRANSTYPE::N) << std::endl;
    std::cout << "\n\n mat3T*vec2" << TensorOp::matmul(mat3,vec2,TRANSTYPE::T) << std::endl;
    std::cout << "\n\n mat1 *mat2" << TensorOp::matmul(mat1,mat2) << std::endl;
    std::cout << "\n\n mat3 *mat2T" << TensorOp::matmul(mat3,mat2,TRANSTYPE::N,TRANSTYPE::T) << std::endl;
*/
    ptr_comm->finalize();
    return 0;
}