#pragma once
#include "Map.hpp"

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
    /*
    // array -> array
	const std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index);
	const std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index);
	// size_t -> array
	const std::array<size_t, dimension> get_global_array_index(const size_t local_index);
	const std::array<size_t, dimension> get_local_array_index (const size_t global_index);
    */
	// size_t -> size_t
	const size_t get_global_index(const size_t local_index);
	const size_t get_local_index(const size_t global_index);
    /*
    // array -> size_t
    connst size_t get_global_index(const std::array<size_t, dimension> local_index);
	const size_t get_local_index(const std::array<size_t, dimension> global_index);
    */
private:
    std::array<size_t, dimension> total_size;
    size_t num_global_elements = -1;
    size_t num_my_elements = -1;
    size_t first_global_index = -1;
    size_t* element_size_list;
};

template<size_t dimension>
ContiguousMap<dimension>::ContiguousMap(std::array<size_t,dimension> total_size, Comm *comm) : Map<dimension>(commPtr), total_size(total_size){
    num_global_elements = 1;
    for(size_t i = 0; i < dimension; ++i){
        num_global_elements *= total_size[i];
    }
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

template <size_t dimension>
inline const size_t ContiguousMap<dimension>::get_global_index(const size_t local_index){
    return first_global_index + local_index;
}
template <size_t dimension>
inline const size_t ContiguousMap<dimension>::get_local_index(const size_t global_index){
    size_t local_index = global_index - first_global_index;
    if(local_index < 0 || local_index > num_my_elements-1){
        std::cout << "invalid global index in rank = " << comm->get_rank() << std::endl;
        return -1;
    }
    return local_index;
}
}