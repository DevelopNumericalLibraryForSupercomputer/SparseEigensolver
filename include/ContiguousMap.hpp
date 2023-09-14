#pragma once
#include "Comm.hpp"
#include "Map.hpp"
#include "Utility_include.hpp"
namespace TH{
template<size_t dimension>
class ContiguousMap: public Map<dimension>{
public:
    ContiguousMap(std::array<size_t, dimension> total_size, Comm& comm); 
    // tensor will only be sliced alog the slice dimension
    // array -> array
    const std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index, size_t slice_dimension);
    const std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index, size_t slice_dimension);
    // size_t -> array
    const std::array<size_t, dimension> get_global_array_index(const size_t local_index, size_t slice_dimension);
    const std::array<size_t, dimension> get_local_array_index (const size_t global_index, size_t slice_dimension);
    // array -> size_t
    const size_t get_global_index(const std::array<size_t, dimension> local_index, size_t slice_dimension);
    const size_t get_local_index(const std::array<size_t, dimension> global_index, size_t slice_dimension);
    // size_t -> size_t
    const size_t get_global_index(const size_t local_index, size_t slice_dimension);
    const size_t get_local_index(const size_t global_index, size_t slice_dimension);
    
private:
    /* contiguous map of the given index.
    num_global_elements / comm.get_world_size() = 98/4 = 24
    num_global_elements % comm.get_world_size() = 98%4 = 2
    rank = 0 : 0 ~ 24   
    rank = 1 : 25 ~ 49 
    rank = 2 : 50 ~ 74 
    rank = 3 : 75 ~ 97 (25-2)
    */
    size_t calculate_chunk_size(size_t num_global_index);
    size_t local_to_global(size_t num_global_index, size_t local_index);
    size_t global_to_local(size_t num_global_index, size_t global_index);
    /* vectorized tensor
    i + ni*j + ni*nj*k+...
    example: rank 3 tensor T_234
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
    ...
    1, 2, 2 = 17
    ...
    1, 2, 3 = 23
    */
    size_t whole_tensor_to_array_index(std::array<size_t, dimension> tensor_index);
    size_t sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension);
    //size_t tensor_to_array_index(std::array<size_t, dimension> tensor_index);
    std::array<size_t, dimension> array_to_whole_tensor_index(size_t array_index);
    std::array<size_t, dimension> array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension);
    
    
};

template<size_t dimension>
ContiguousMap<dimension>::ContiguousMap(std::array<size_t,dimension> total_size, Comm& comm) : Map<dimension>(comm){
    this->tensor_total_size = total_size;
    cumprod(total_size, this->tensor_total_size_mult);
    //this->num_global_elements = this->tensor_total_size_mult[dimension];
    /*
    const size_t rank = comm.get_rank();
    const size_t world_size = comm.get_world_size();
    size_t quotient  = this->num_global_elements / world_size;
    size_t remainder = this->num_global_elements % world_size;
    this->element_size_list = new size_t[world_size];
    this->first_my_global_index = 0;
    for(size_t i = 0; i < world_size; ++i){
        this->element_size_list[i] = ( i>=remainder ) ? quotient : quotient + 1;
        if(i<rank) {
            this->first_my_global_index += this->element_size_list[i];
        }
    }
    this->num_my_elements = this->element_size_list[rank];
    */

};

// array -> array
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const std::array<size_t, dimension> local_index, size_t slice_dimension){
    std::array<size_t, dimension> global_index = local_index;
    global_index[slice_dimension] = local_to_global(tensor_total_size[slice_dimension], local_index[slice_dimension]);
    return global_index;
}
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const std::array<size_t, dimension> global_index, size_t slice_dimension){
    std::array<size_t, dimension> local_index = global_index;
    local_index[slice_dimension] = global_to_local(tensor_total_size[slice_dimension], global_index[slice_dimension]);
    return local_index;
}

// size_t -> array
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const size_t local_index, size_t slice_dimension){
    std::array<size_t, dimension> local_array_index = array_to_sliced_tensor_index(local_index, slice_dimension);
    return get_global_array_index(local_array_index,slice_dimension);
}
template <size_t dimension>
const std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const size_t global_index, size_t slice_dimension){
    std::array<size_t, dimension> global_array_index = array_to_whole_tensor_index(global_index);
    return get_local_array_index(global_array_index,slice_dimension);
}

// array -> size_t
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_global_index(const std::array<size_t, dimension> local_index, size_t slice_dimension){
    std::array<size_t,dimension> global_array_index = get_global_array_index(local_index, slice_dimension);
    return whole_tensor_to_array_index(global_array_index);
    
}
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_local_index(const std::array<size_t, dimension> global_index, size_t slice_dimension){
    std::array<size_t,dimension> local_array_index = get_local_array_index(global_index, slice_dimension);
    return sliced_tensor_to_array_index(local_array_index,slice_dimension);

}

// size_t -> size_t
template<size_t dimension>
const size_t ContiguousMap<dimension>::get_global_index(const size_t local_index, size_t slice_dimension){
    // return this->first_my_global_index + local_index;
    exit(-1);
}
template <size_t dimension>
const size_t ContiguousMap<dimension>::get_local_index(const size_t global_index, size_t slice_dimension){
    /*
    size_t local_index = global_index - this->first_my_global_index;
    if(local_index < 0 || local_index > this->num_my_elements-1){
        std::cout << "invalid global index in rank = " << this->comm.get_rank() << std::endl;
        exit(-1);
    }
    return local_index;
    */
    exit(-1);

}


// Calculate the chunk size for each thread
template <size_t dimension>
size_t ContiguousMap<dimension>::calculate_chunk_size(size_t num_global_index){
    size_t num_threads = this->comm.get_world_size();
    size_t chunk_size = num_global_index / num_threads;
    size_t remainder = num_global_index % num_threads;
    // Distribute any remaining elements
    if(remainder>0) { chunk_size++; }
    return chunk_size;
}

// Convert local index to global index
template <size_t dimension>
size_t ContiguousMap<dimension>::local_to_global(size_t num_global_index, size_t local_index){
    size_t chunk_size = calculate_chunk_size(num_global_index, this->comm.get_world_size());
    size_t chunk_start = chunk_size * this->comm.get_rank();
    size_t chunk_end = std::min(chunk_start + chunk_size, num_global_index);
    assert (local_index < (chunk_end - chunk_start));
    return chunk_start + local_index;
}

// Convert global index to local index
template <size_t dimension>
size_t ContiguousMap<dimension>::global_to_local(size_t num_global_index, size_t global_index) {
    size_t chunk_size = calculate_chunk_size(num_global_index, this->comm.get_world_size());
    size_t chunk_start = chunk_size * this->comm.get_rank();
    size_t chunk_end = std::min(chunk_start + chunk_size, num_global_index);
    assert(global_index >= chunk_start);
    assert(global_index < chunk_end);
    return global_index - chunk_start;
}

template <size_t dimension>
size_t ContiguousMap<dimension>::whole_tensor_to_array_index(std::array<size_t, dimension> tensor_index){
    size_t return_index = 0;
    for(size_t dim = 0; dim < dimension; ++dim){
        assert(tensor_index[dim] < this->tensor_total_size[dim]);
        return_index += tensor_index[dim] * tensor_total_size_mult[dim];
    }
    return return_index;
}

template <size_t dimension>
size_t ContiguousMap<dimension>::sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension){
    std::array<size_t, dimesion> sliced_tensor_total_size = tensor_total_size;
    sliced_tensor_total_size[slice_dimension] = calculate_chunk_size(tensor_total_size[slice_dimension]);

    size_t return_index = 0;
    size_t multiplier = 1;
    for(size_t dim = 0; dim < dimension; ++dim){
        assert(tensor_index[dim] < sliced_tensor_total_size[dim]);
        return_index += tensor_index[dim] * multiplier;
        multiplier *= sliced_tensor_total_size[dim]
    }
    return return_index;
}

template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::array_to_whole_tensor_index(size_t array_index){
    assert(array_index < this->tensor_total_size_mult[dimension]);
    std::array<size_t, dimension> return_index;
    for(size_t dim = 0; dim <dimension; ++dim){
        return_index[dim] = array_index % this->tensor_total_size[dim];
        array_index /= this->tensor_total_size[dim];
    }
    return return_index;
}

template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension){
    std::array<size_t, dimesion> sliced_tensor_total_size = tensor_total_size;
    sliced_tensor_total_size[slice_dimension] = calculate_chunk_size(tensor_total_size[slice_dimension]);

    std::array<size_t, dimension> return_index;
    for(size_t dim = 0; dim <dimension; ++dim){
        return_index[dim] = array_index % sliced_tensor_total_size[dim];
        array_index /= sliced_tensor_total_size[dim];
    }
    return return_index;

}

}