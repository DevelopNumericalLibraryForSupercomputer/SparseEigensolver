#pragma once
#include "Map.hpp"
namespace SE{
template<size_t dimension>
class ContiguousMap: public Map<dimension>{
public:
    ContiguousMap(std::array<size_t, dimension> total_size) : Map<dimension>(total_size) {};
    // tensor will only be sliced along the slice dimension
    // array -> array
    std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank, size_t world_size);
    std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank, size_t world_size);
    // size_t -> array
    std::array<size_t, dimension> get_global_array_index(const size_t local_index, size_t slice_dimension, size_t rank, size_t world_size);
    std::array<size_t, dimension> get_local_array_index (const size_t global_index, size_t slice_dimension, size_t rank, size_t world_size);
    // array -> size_t
    size_t get_global_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank, size_t world_size);
    size_t get_local_index(const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank, size_t world_size);
    // size_t -> size_t
    size_t get_global_index(const size_t local_index, size_t slice_dimension, size_t rank, size_t world_size);
    size_t get_local_index(const size_t global_index, size_t slice_dimension, size_t rank, size_t world_size);

    size_t get_my_rank_from_global_index(const size_t global_index, size_t slice_dimension, size_t world_size);
    size_t calculate_chunk_size(size_t num_global_index, size_t world_size);
    
private:
    /* contiguous map of the given index.
    num_global_elements / comm.get_world_size() +1 = 98/4 = 24
    num_global_elements % comm.get_world_size() = 98%4 = 2
    rank = 0 : 0 ~ 23   
    rank = 1 : 24 ~ 47 
    rank = 2 : 48 ~ 72 
    rank = 3 : 73 ~ 97 (25+2)
    */
    size_t local_to_global(size_t num_global_index, size_t local_index, size_t rank, size_t world_size);
    size_t global_to_local(size_t num_global_index, size_t global_index, size_t rank, size_t world_size);
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
    size_t sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank, size_t world_size);
    std::array<size_t, dimension> array_to_whole_tensor_index(size_t array_index);
    std::array<size_t, dimension> array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension, size_t rank, size_t world_size);
};

// array -> array
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> global_index = local_index;
    global_index[slice_dimension] = local_to_global(this->tensor_total_size[slice_dimension], local_index[slice_dimension], rank, world_size);
    return global_index;
}
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> local_index = global_index;
    local_index[slice_dimension] = global_to_local(this->tensor_total_size[slice_dimension], global_index[slice_dimension], rank, world_size);
    return local_index;
}

// size_t -> array
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const size_t local_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> local_array_index = array_to_sliced_tensor_index(local_index, slice_dimension, rank, world_size);
    return get_global_array_index(local_array_index,slice_dimension, rank, world_size);
}
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const size_t global_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> global_array_index = array_to_whole_tensor_index(global_index);
    return get_local_array_index(global_array_index,slice_dimension, rank, world_size);
}

// array -> size_t
template <size_t dimension>
size_t ContiguousMap<dimension>::get_global_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t,dimension> global_array_index = get_global_array_index(local_index, slice_dimension, rank, world_size);
    return whole_tensor_to_array_index(global_array_index);
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_local_index(const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t,dimension> local_array_index = get_local_array_index(global_index, slice_dimension, rank, world_size);
    return sliced_tensor_to_array_index(local_array_index, slice_dimension, rank, world_size);
}

// size_t -> size_t
template <size_t dimension>
size_t ContiguousMap<dimension>::get_global_index(const size_t local_index, size_t slice_dimension, size_t rank, size_t world_size){
    return get_global_index(array_to_sliced_tensor_index(local_index, slice_dimension, rank, world_size), slice_dimension, rank, world_size);
}
template <size_t dimension>
size_t ContiguousMap<dimension>::get_local_index(const size_t global_index, size_t slice_dimension, size_t rank, size_t world_size){
    return get_local_index(array_to_whole_tensor_index(global_index),slice_dimension, rank, world_size);   
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_my_rank_from_global_index(const size_t global_index, size_t slice_dimension, size_t world_size){
    size_t chunk_size = calculate_chunk_size(this->tensor_total_size[slice_dimension], world_size);
    return global_index / chunk_size;
}

// Calculate the chunk size
template <size_t dimension>
size_t ContiguousMap<dimension>::calculate_chunk_size(size_t num_global_index, size_t world_size){
    return num_global_index / world_size;
}

// Convert local index to global index
template <size_t dimension>
size_t ContiguousMap<dimension>::local_to_global(size_t num_global_index, size_t local_index, size_t rank, size_t world_size){
    size_t chunk_size = calculate_chunk_size(num_global_index, world_size);
    size_t chunk_start = chunk_size * rank;
    size_t chunk_endp1 = world_size-1 == rank ? num_global_index : chunk_start+chunk_size;
    //std::cout << "local_to_global, rank " << this->comm.get_rank() << " " << num_global_index << " " << local_index << " " << chunk_size << " " << chunk_start << " " << chunk_endp1 << std::endl;
    assert (local_index < (chunk_endp1 - chunk_start));
    return chunk_start + local_index;
}

// Convert global index to local index
template <size_t dimension>
size_t ContiguousMap<dimension>::global_to_local(size_t num_global_index, size_t global_index, size_t rank, size_t world_size) {
    size_t chunk_size = calculate_chunk_size(num_global_index, world_size);
    size_t chunk_start = chunk_size * rank;
    size_t chunk_end = world_size-1 == rank ? num_global_index-1 : chunk_start+chunk_size-1;
    assert(global_index >= chunk_start);
    assert(global_index <= chunk_end);
    return global_index - chunk_start;
}

template <size_t dimension>
size_t ContiguousMap<dimension>::whole_tensor_to_array_index(std::array<size_t, dimension> tensor_index){
    size_t return_index = 0;
    for(size_t dim = 0; dim < dimension; ++dim){
        assert(tensor_index[dim] < this->tensor_total_size[dim]);
        return_index += tensor_index[dim] * this->tensor_total_size_mult[dim];
    }
    return return_index;
}

template <size_t dimension>
size_t ContiguousMap<dimension>::sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> sliced_tensor_total_size = this->tensor_total_size;
    size_t sliced_index_size = calculate_chunk_size(this->tensor_total_size[slice_dimension], world_size);
    if(world_size-1 == rank){
        sliced_index_size = this->tensor_total_size[slice_dimension] - sliced_index_size * rank;
    }
    sliced_tensor_total_size[slice_dimension] = sliced_index_size;

    size_t return_index = 0;
    size_t multiplier = 1;
    for(size_t dim = 0; dim < dimension; ++dim){
        assert(tensor_index[dim] < sliced_tensor_total_size[dim]);
        return_index += tensor_index[dim] * multiplier;
        multiplier *= sliced_tensor_total_size[dim];
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
std::array<size_t, dimension> ContiguousMap<dimension>::array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension, size_t rank, size_t world_size){
    std::array<size_t, dimension> sliced_tensor_total_size = this->tensor_total_size;
    size_t sliced_index_size = calculate_chunk_size(this->tensor_total_size[slice_dimension], world_size);
    if(world_size-1 == rank){
        sliced_index_size = this->tensor_total_size[slice_dimension] - sliced_index_size * rank;
    }
    sliced_tensor_total_size[slice_dimension] = sliced_index_size;

    std::array<size_t, dimension> return_index;
    for(size_t dim = 0; dim <dimension; ++dim){
        return_index[dim] = array_index % sliced_tensor_total_size[dim];
        array_index /= sliced_tensor_total_size[dim];
    }
    return return_index;
}

}
