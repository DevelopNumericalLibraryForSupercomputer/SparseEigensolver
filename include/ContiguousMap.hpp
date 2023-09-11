#pragma once
#include "Map.hpp"
#include "Utility.hpp"
namespace TensorHetero{
template<size_t dimension>
class ContiguousMap: public Map<dimension>{
public:
    ContiguousMap(std::array<size_t, dimension> total_size, Comm *comm); 
    /*
    num_global_elements = 98개, world_size = 4일때
    num_global_elements / comm->get_world_size() = 98/4 = 24
    num_global_elements % comm->get_world_size() = 98%4 = 2
    
    rank = 0 : 0 ~ 24   
    rank = 1 : 25 ~ 49 
    rank = 2 : 50 ~ 73 
    rank = 3 : 74 ~ 97 
    
    local index = 4, rank = 2 --> global index = 53
    */
    
    // array -> array
    const std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index);
    const std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index);
    // size_t -> array
    const std::array<size_t, dimension> get_global_array_index(const size_t local_index);
    const std::array<size_t, dimension> get_local_array_index (const size_t global_index);
    // array -> size_t
    const size_t get_global_index(const std::array<size_t, dimension> local_index);
    const size_t get_local_index(const std::array<size_t, dimension> global_index);
    // size_t -> size_t
    const size_t get_global_index(const size_t local_index);
    const size_t get_local_index(const size_t global_index);
    
private:
    std::array<size_t, dimension> total_size;
    std::array<size_t, dimension+1> total_size_mult;
    size_t num_global_elements = -1;
    size_t num_my_elements = -1;
    size_t first_global_index = -1;
    size_t* element_size_list;

    size_t tensor_to_array_index(std::array<size_t, dimension> tensor_index) const;
    std::array<size_t, dimension> array_to_tensor_index(size_t array_index) const;
    
};

template<size_t dimension>
ContiguousMap<dimension>::ContiguousMap(std::array<size_t,dimension> total_size, Comm *comm) : Map<dimension>(commPtr), total_size(total_size){
    cumprod(total_size, total_size_mult);
    num_global_elements = total_size_mult[dimension];
    cosnt size_t rank = comm->get_rank();
    cosnt size_t world_size = comm->get_world_size();

    size_t quotient  = num_global_elements / world_size;
    size_t remainder = num_global_elements % world_size;
    element_size_list = new size_t[world_size];
    first_index_my_elements = 0;
    for(size_t i = 0; i < world_size; ++i){
        element_size_list[i] = ( i>reminder ) ? quotient : quotient + 1;
        if(i<rank) {
            first_global_index += element_size_list[i];
        }
    }
    num_my_elements = element_size_list[rank];

};

// array -> array
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const std::array<size_t, dimension> local_index){
    /* local array not defined*/
    std::cout << "local array index is not defined" << std::endl;
    exit(-1);
}
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const std::array<size_t, dimension> global_index){
    /* local array not defined*/
    std::cout << "local array index is not defined" << std::endl;
    exit(-1);
}

// size_t -> array
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const size_t local_index){
    size_t global_index = get_global_index(local_index);
    return array_to_tensor_index(global_index);
}
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const size_t global_index){
    /* local array not defined*/
    std::cout << "local array index is not defined" << std::endl;
    exit(-1);
}

// array -> size_t
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_global_index(const std::array<size_t, dimension> local_index){
    /* local array not defined*/
    std::cout << "local array index is not defined" << std::endl;
    exit(-1);
}
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_local_index(const std::array<size_t, dimension> global_index){
    size_t global_1D_index = tensor_to_array_index(global_index);
    return get_local_index(global_1D_index);
}

// size_t -> size_t
template<size_t dimension>
const size_t ContiguousMap<dimension>::get_global_index(const size_t local_index){
    return first_global_index + local_index;
}
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_local_index(const size_t global_index){
    size_t local_index = global_index - first_global_index;
    if(local_index < 0 || local_index > num_my_elements-1){
        std::cout << "invalid global index in rank = " << comm->get_rank() << std::endl;
        return -1;
    }
    return local_index;

}

/* example
i, j, k = index
0, 0, 0 = 0
1, 0, 0 = 1
0, 1, 0 = 2
1, 1, 0 = 3
0, 2, 0 = 4
1, 2, 0 = 5
0, 0, 1 = 6
1, 0, 1 = 7
0, 1, 1 = 8
1, 1, 1 = 9
0, 2, 1 = 10
1, 2, 1 = 11
0, 0, 2 = 12
1, 0, 2 = 13
0, 1, 2 = 14
1, 1, 2 = 15
0, 2, 2 = 16
1, 2, 2 = 17
0, 0, 3 = 18
1, 0, 3 = 19
0, 1, 3 = 20
1, 1, 3 = 21
0, 2, 3 = 22
1, 2, 3 = 23
*/

template <size_t dimension>
size_t ContiguousMap<dimension>::tensor_to_array_index(std::array<size_t, dimension> tensor_index) const {
    size_t return_index = 0;
    size_t multiplier = 1;
    for(size_t dim = 0; dim < N; ++dim){
        assert(tensor_index[dim] >= total_size[dim]);
        return_index += tensor_index[dim] * multiplier;
        multiplier *= total_size[dim];
    }
    return return_index;
}

template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::array_to_tensor_index(size_t array_index) const {
    assert(array_index < num_global_elements);
    size_t input_index = array_index;
    std::array<size_t, dimension> return_index;
    for(size_t dim = 0; dim <N; ++dim){
        return_index[dim] = input_index % total_size[dim];
        input_index /= total_size[dim];
    }
    return return_index;

}

}