#pragma once
#include <iostream>
#include <array>
#include <cassert>
//#include "Comm_include.hpp"
#include "Comm.hpp"
#include "Utility.hpp"
namespace SE{
template<size_t dimension>
class Map{
public:
	Map(){};

	Map(std::array<size_t, dimension> total_size, size_t world_size);
	//Map(std::array<size_t, dimension> total_size, size_t world_size, size_t slice_dimension) = 0;

	size_t get_sliced_dim(){return sliced_dimension;}
	bool get_is_sliced(){return is_sliced;}
	virtual void redistribute(bool slice, size_t slice_dimension) = 0; // slice == true : partition , false : merge
 
	virtual size_t get_start_my_global_index(size_t slice_dimension, size_t rank) = 0;
	virtual size_t get_end_my_global_index(size_t slice_dimension, size_t rank) = 0;
	//virtual size_t get_end_my_local_index(size_t slice_dimension, size_t rank) = 0;
	virtual size_t get_my_partition_size(size_t slice_dimension, size_t rank) = 0;
	virtual size_t* get_partition_size_array(size_t slice_dimension) = 0;
	virtual size_t* get_start_global_index_array(size_t slice_dimension) = 0;

	virtual size_t get_my_rank_from_global_index(const size_t global_index, size_t slice_dimension) = 0;
	//virtual size_t calculate_chunk_size(size_t num_global_index, size_t world_size) = 0;
	
	// array -> array
	virtual std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank) = 0;
	virtual std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank) = 0;
	// size_t -> array
	virtual std::array<size_t, dimension> get_global_array_index(const size_t local_index, size_t slice_dimension, size_t rank) = 0;
	virtual std::array<size_t, dimension> get_local_array_index (const size_t global_index, size_t slice_dimension, size_t rank) = 0;
	// array -> size_t
	virtual size_t get_global_index(const std::array<size_t, dimension> local_index, size_t slice_dimension, size_t rank) = 0;
	virtual size_t get_local_index(const std::array<size_t, dimension> global_index, size_t slice_dimension, size_t rank) = 0;
	// size_t -> size_t
	virtual size_t get_global_index(const size_t local_index, size_t slice_dimension, size_t rank) = 0;
	virtual size_t get_local_index(const size_t global_index, size_t slice_dimension, size_t rank) = 0;

	/*
	const size_t get_num_global_elements(){return num_global_elements;};
	const size_t get_num_my_elements(){return num_my_elements;};
	const size_t get_first_my_global_index(){return first_my_global_index;};
	const size_t* get_element_size_list(){return element_size_list;};
	*/
protected:
	std::array<size_t, dimension> tensor_total_size;
    std::array<size_t, dimension+1> tensor_total_size_mult;
	size_t world_size;

	bool is_sliced;
	size_t sliced_dimension; // if is_sliced is false, sliced_dimension will become 0
	size_t* partition_size;
	size_t* start_global_index;

	/*
    size_t num_global_elements = -1;
    size_t num_my_elements = -1;
    size_t first_my_global_index = -1;
    size_t* element_size_list;
	*/

};

template <size_t dimension>
inline Map<dimension>::Map(std::array<size_t, dimension> total_size, size_t world_size){
	this->tensor_total_size = total_size;
    cumprod(total_size, this->tensor_total_size_mult);
	this->world_size = world_size;
	is_sliced = false;
}
/*
template <size_t dimension>
inline Map<dimension>::Map(std::array<size_t, dimension> total_size, size_t world_size, size_t slice_dimension){
	assert(slice_dimension < total_size.size());
	this->tensor_total_size = total_size;
    cumprod(total_size, this->tensor_total_size_mult);
	this->world_size = world_size;
	redistribute(true, slice_dimension);
}

template <size_t dimension>
void Map<dimension>::redistribute(bool slice, size_t slice_dimension){
	
}
*/

}
