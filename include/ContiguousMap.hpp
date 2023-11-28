#pragma once
#include "Map.hpp"
namespace SE{
template<size_t dimension>
class ContiguousMap: public Map<dimension>{
public:
    ContiguousMap(std::array<size_t, dimension> total_size, size_t world_size) : Map<dimension>(total_size, world_size) {};
    ContiguousMap(std::array<size_t, dimension> total_size, size_t world_size, size_t slice_dimension);// : Map<dimension>(total_size, world_size, slice_dimension) {};
    // tensor will only be sliced along the slice dimension

	void redistribute(bool slice, size_t slice_dimension); // slice == true : partition , false : merge
 
	size_t get_start_my_global_index(size_t rank);
	size_t get_end_my_global_index(size_t rank);
	size_t get_my_partition_size(size_t rank);        //partition size of sliced dimension
    size_t get_my_partitioned_data_size(size_t rank); //partition size of total data
	size_t* get_partition_size_array();
	size_t* get_start_global_index_array();

    size_t get_my_rank_from_global_index(const size_t global_index);
    size_t get_my_rank_from_global_index(const std::array<size_t, dimension> global_index);
    //size_t calculate_chunk_size(size_t num_global_index);



    // array -> array
    std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index, size_t rank);
    std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index, size_t rank);
    // size_t -> array
    std::array<size_t, dimension> get_global_array_index(const size_t local_index, size_t rank);
    std::array<size_t, dimension> get_local_array_index (const size_t global_index, size_t rank);
    // array -> size_t
    size_t get_global_index(const std::array<size_t, dimension> local_index, size_t rank);
    size_t get_local_index(const std::array<size_t, dimension> global_index, size_t rank);
    // size_t -> size_t
    size_t get_global_index(const size_t local_index, size_t rank);
    size_t get_local_index(const size_t global_index, size_t rank);

    
private:
    /* contiguous map of the given index.
    num_global_elements / comm.get_world_size() +1 = 98/4 = 24
    num_global_elements % comm.get_world_size() = 98%4 = 2
    rank = 0 : 0 ~ 23   
    rank = 1 : 24 ~ 47 
    rank = 2 : 48 ~ 72 
    rank = 3 : 73 ~ 97 (25+2)
    */
    //size_t local_to_global(size_t num_global_index, size_t local_index, size_t rank);
    //size_t global_to_local(size_t num_global_index, size_t global_index, size_t rank);
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
    /*
    size_t whole_tensor_to_array_index(std::array<size_t, dimension> tensor_index);
    size_t sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank);
    std::array<size_t, dimension> array_to_whole_tensor_index(size_t array_index);
    std::array<size_t, dimension> array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension, size_t rank);
    */
    size_t global_tensor_index_flatten(std::array<size_t, dimension> tensor_index);
    size_t sliced_tensor_index_flatten(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank);
    std::array<size_t, dimension> global_index_unflatten(size_t array_index);
    std::array<size_t, dimension> sliced_index_unflatten(size_t array_index, size_t slice_dimension, size_t rank);

};


template <size_t dimension>
ContiguousMap<dimension>::ContiguousMap(std::array<size_t, dimension> total_size, size_t world_size, size_t slice_dimension){
	assert(slice_dimension < total_size.size());
	this->tensor_total_size = total_size;
    cumprod(total_size, this->tensor_total_size_mult);
	this->world_size = world_size;
    //std::cout << "constructor : " << world_size <<  ", slice dim " << slice_dimension << std::endl;
    if(this->world_size >1){
        redistribute(true, slice_dimension);
    }
    else{
        this->is_sliced = false;
        this->sliced_dimension = 0;
        /*
	    this->partition_size = new size_t[1];
	    this->start_global_index = new size_t[1];
	    this->partition_size[0] = total_size[0];
	    this->start_global_index[0] = 0;
        */
    }
}

template <size_t dimension>
void ContiguousMap<dimension>::redistribute(bool slice, size_t slice_dimension){
    //std::cout << "redistribute " << std::endl;
    if(slice && this->world_size>1){
        this->is_sliced = true;
        this->sliced_dimension = slice_dimension;
        //size_t num_global_elements =this->tensor_total_size[slice_dimension];

        this->partition_size = new size_t[this->world_size];
        this->start_global_index = new size_t[this->world_size];

        size_t chunk_size = this->tensor_total_size[slice_dimension] / this->world_size;
        this->start_global_index[0] = 0;
        for(size_t i=0;i<this->world_size-1;i++){
            this->partition_size[i] = chunk_size;
            this->start_global_index[i+1] = this->start_global_index[i] + this->partition_size[i];
        }
        //this->start_global_index[this->world_size-1] = this->tensor_total_size[slice_dimension];
        this->partition_size[this->world_size-1] = this->tensor_total_size[slice_dimension] - this->start_global_index[this->world_size-1];
        //std::cout << "redistribute " << slice_dimension << " "  << chunk_size << " " << this->world_size << " " << this->partition_size[0] << " " << this->start_global_index[0] << " " << this->tensor_total_size[0] << std::endl;
        //std::cout << "redistribute " << slice_dimension << " "  << chunk_size << " " << this->world_size << " " << this->partition_size[1] << " " << this->start_global_index[1] << " " << this->tensor_total_size[1] << std::endl;
    }
    else{
        this->is_sliced = false;
        this->sliced_dimension = 0;
        std::free(this->partition_size);
        std::free(this->start_global_index);
    }
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_start_my_global_index( size_t rank){
    if(this->is_sliced){
        return this->start_global_index[rank];
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}
template <size_t dimension>
size_t ContiguousMap<dimension>::get_end_my_global_index(size_t rank){
    if(this->is_sliced){
        if(rank < this->world_size){
            return this->start_global_index[rank+1];
        }
        else{
            return this->world_size;
        }
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}
template <size_t dimension>
size_t ContiguousMap<dimension>::get_my_partition_size(size_t rank){
    if(this->is_sliced){
        return this->partition_size[rank];
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}
template <size_t dimension>
size_t ContiguousMap<dimension>::get_my_partitioned_data_size(size_t rank){
    if(this->is_sliced){
        return this->tensor_total_size_mult[dimension] * get_my_partition_size(rank) / this->tensor_total_size[this->sliced_dimension];
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}
template <size_t dimension>
size_t *ContiguousMap<dimension>::get_partition_size_array()
{
    if(this->is_sliced){
        return this->partition_size;
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}
template <size_t dimension>
size_t* ContiguousMap<dimension>::get_start_global_index_array(){
    if(this->is_sliced){
        return this->start_global_index;
    }
    else{
        std::cout << "not partitioned. wrong index!" << std::endl;
        exit(-1);
    }
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_my_rank_from_global_index(const size_t global_index){
    if(this->is_sliced){
        size_t chunk_size = this->tensor_total_size[this->sliced_dimension] / this->world_size;
        size_t my_rank = global_index / chunk_size;
        if(my_rank >= this->world_size) my_rank = this->world_size-1;
        return my_rank;
    }
    else{
        return 0;
    }
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_my_rank_from_global_index(const std::array<size_t, dimension> global_index){
    if(this->is_sliced){
        return get_my_rank_from_global_index(global_index[this->sliced_dimension]);
    }
    else{
        return 0;
    }
}

// array -> array
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const std::array<size_t, dimension> local_index, size_t rank){
    std::array<size_t, dimension> global_index = local_index;
    if(this->is_sliced){

        //if(!(this->partition_size[rank] > local_index[this->sliced_dimension])){
        //    std::cout << "WRONG : " << rank << ' ' << this->partition_size[rank] << ' ' << local_index[this->sliced_dimension] << std::endl;
        //    exit(-1);
        //}
        assert(this->partition_size[rank] > local_index[this->sliced_dimension]);
        global_index[this->sliced_dimension] = this->start_global_index[rank] + local_index[this->sliced_dimension];
    }
    return global_index;
}
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const std::array<size_t, dimension> global_index, size_t rank){
    std::array<size_t, dimension> local_index = global_index;
    
    if(this->is_sliced){
        local_index[this->sliced_dimension] = global_index[this->sliced_dimension] - this->start_global_index[rank];
        //std::cout << "here3 " << local_index[this->sliced_dimension] << ' ' <<  global_index[this->sliced_dimension] <<' ' << this->start_global_index[rank] << std::endl;
    }
    return local_index;
}

// size_t -> array
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_global_array_index(const size_t local_index, size_t rank){
    std::array<size_t, dimension> local_array_index = sliced_index_unflatten(local_index, this->sliced_dimension, rank);
    //std::cout << "local : " << local_index << " , local-array : " << local_array_index[0] << " " << local_array_index[1] << ", " << rank << std::endl;
    return get_global_array_index(local_array_index, rank);
}
template <size_t dimension>
std::array<size_t, dimension> ContiguousMap<dimension>::get_local_array_index(const size_t global_index, size_t rank){
    std::array<size_t, dimension> global_array_index = global_index_unflatten(global_index);
    return get_local_array_index(global_array_index, rank);
}

// array -> size_t
template <size_t dimension>
size_t ContiguousMap<dimension>::get_global_index(const std::array<size_t, dimension> local_index, size_t rank){
    std::array<size_t,dimension> global_array_index = get_global_array_index(local_index, rank);
    return global_tensor_index_flatten(global_array_index);
}

template <size_t dimension>
size_t ContiguousMap<dimension>::get_local_index(const std::array<size_t, dimension> global_index, size_t rank){
    std::array<size_t,dimension> local_array_index = get_local_array_index(global_index, rank);
    //std::cout << "here" << std::endl;
    return sliced_tensor_index_flatten(local_array_index, this->sliced_dimension, rank);
}

// size_t -> size_t
template <size_t dimension>
size_t ContiguousMap<dimension>::get_global_index(const size_t local_index, size_t rank){
    return get_global_index(sliced_index_unflatten(local_index, this->sliced_dimension, rank), rank);
}
template <size_t dimension>
size_t ContiguousMap<dimension>::get_local_index(const size_t global_index, size_t rank){
    return get_local_index(global_index_unflatten(global_index), rank);   
}


/*
// Convert local index to global index
template <size_t dimension>
size_t ContiguousMap<dimension>::local_to_global(size_t num_global_index, size_t local_index, size_t rank){
    size_t chunk_size = calculate_chunk_size(num_global_index, world_size);
    size_t chunk_start = chunk_size * rank;
    size_t chunk_endp1 = world_size-1 == rank ? num_global_index : chunk_start+chunk_size;
    //std::cout << "local_to_global, rank " << this->comm.get_rank() << " " << num_global_index << " " << local_index << " " << chunk_size << " " << chunk_start << " " << chunk_endp1 << std::endl;
    assert (local_index < (chunk_endp1 - chunk_start));
    return chunk_start + local_index;
}


// Convert global index to local index
template <size_t dimension>
size_t ContiguousMap<dimension>::global_to_local(size_t num_global_index, size_t global_index, size_t rank) {
    size_t chunk_size = calculate_chunk_size(num_global_index, world_size);
    size_t chunk_start = chunk_size * rank;
    size_t chunk_end = world_size-1 == rank ? num_global_index-1 : chunk_start+chunk_size-1;
    assert(global_index >= chunk_start);
    assert(global_index <= chunk_end);
    return global_index - chunk_start;
}
*/
template <size_t dimension>
//size_t ContiguousMap<dimension>::whole_tensor_to_array_index(std::array<size_t, dimension> tensor_index){
size_t ContiguousMap<dimension>::global_tensor_index_flatten(std::array<size_t, dimension> tensor_index){
    size_t return_index = 0;
    for(size_t dim = 0; dim < dimension; ++dim){
        assert(tensor_index[dim] < this->tensor_total_size[dim]);
        return_index += tensor_index[dim] * this->tensor_total_size_mult[dim];
    }
    return return_index;
}

template <size_t dimension>
//size_t ContiguousMap<dimension>::sliced_tensor_to_array_index(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank){
size_t ContiguousMap<dimension>::sliced_tensor_index_flatten(std::array<size_t, dimension> tensor_index, size_t slice_dimension, size_t rank){    
    if(!this->is_sliced || this->sliced_dimension != slice_dimension){
        return global_tensor_index_flatten(tensor_index);
    }
    else{
        std::array<size_t, dimension> sliced_tensor_total_size = this->tensor_total_size;
        /*size_t sliced_index_size = calculate_chunk_size(this->tensor_total_size[slice_dimension], world_size);
        if(world_size-1 == rank){
            sliced_index_size = this->tensor_total_size[slice_dimension] - sliced_index_size * rank;
        }
        sliced_tensor_total_size[slice_dimension] = sliced_index_size;
        */
        sliced_tensor_total_size[slice_dimension] = this->partition_size[rank];

        size_t return_index = 0;
        size_t multiplier = 1;
        for(size_t dim = 0; dim < dimension; ++dim){
            //std::cout << dim << " " << tensor_index[dim] << " , " << sliced_tensor_total_size[dim] << std::endl;
            assert(tensor_index[dim] < sliced_tensor_total_size[dim]);
            //if(!(tensor_index[dim] < sliced_tensor_total_size[dim])){    
            //std::cout << "WRONG " << dim << " " << tensor_index[dim] << " , " << sliced_tensor_total_size[dim] << std::endl;
            //exit(-1);
            //}
            return_index += tensor_index[dim] * multiplier;
            multiplier *= sliced_tensor_total_size[dim];
        }
        return return_index;
    }
}

template <size_t dimension>
//std::array<size_t, dimension> ContiguousMap<dimension>::array_to_whole_tensor_index(size_t array_index){
std::array<size_t, dimension> ContiguousMap<dimension>::global_index_unflatten(size_t array_index){
    assert(array_index < this->tensor_total_size_mult[dimension]);
    std::array<size_t, dimension> return_index;
    for(size_t dim = 0; dim <dimension; ++dim){
        return_index[dim] = array_index % this->tensor_total_size[dim];
        array_index /= this->tensor_total_size[dim];
    }
    return return_index;
}

template <size_t dimension>
//std::array<size_t, dimension> ContiguousMap<dimension>::array_to_sliced_tensor_index(size_t array_index, size_t slice_dimension, size_t rank){
std::array<size_t, dimension> ContiguousMap<dimension>::sliced_index_unflatten(size_t array_index, size_t slice_dimension, size_t rank){
    if(!this->is_sliced || this->sliced_dimension != slice_dimension){
        return global_index_unflatten(array_index);
    }
    else{
        std::array<size_t, dimension> sliced_tensor_total_size = this->tensor_total_size;
        /*
        size_t sliced_index_size = calculate_chunk_size(this->tensor_total_size[slice_dimension], world_size);
        if(world_size-1 == rank){
            sliced_index_size = this->tensor_total_size[slice_dimension] - sliced_index_size * rank;
        }
        sliced_tensor_total_size[slice_dimension] = sliced_index_size;
        */
        sliced_tensor_total_size[this->sliced_dimension] = this->partition_size[rank];
        std::array<size_t, dimension> return_index;
        for(size_t dim = 0; dim <dimension; ++dim){
            return_index[dim] = array_index % sliced_tensor_total_size[dim];
            array_index /= sliced_tensor_total_size[dim];
        }
        return return_index;
    }
}



}
