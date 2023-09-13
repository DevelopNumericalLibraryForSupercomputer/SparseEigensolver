#pragma once
#include <iostream>
#include <array>
#include "Comm.hpp"

namespace TensorHetero{
template<size_t dimension>
class Map{
public:
	Map(Comm &comm_input): comm(comm_input){};
	// array -> array
	virtual const std::array<size_t, dimension> get_global_array_index(const std::array<size_t, dimension> local_index)  = 0;
	virtual const std::array<size_t, dimension> get_local_array_index (const std::array<size_t, dimension> global_index) = 0;
	// size_t -> array
	virtual const std::array<size_t, dimension> get_global_array_index(const size_t local_index) = 0;
	virtual const std::array<size_t, dimension> get_local_array_index (const size_t global_index) = 0;
	// array -> size_t
	virtual const size_t get_global_index(const std::array<size_t, dimension> local_index) = 0;
	virtual const size_t get_local_index(const std::array<size_t, dimension> global_index) = 0;
	// size_t -> size_t
	virtual const size_t get_global_index(const size_t local_index) = 0;
	virtual const size_t get_local_index(const size_t global_index) = 0;

	const Comm get_comm(){return comm;};
	const size_t get_num_global_elements(){return num_global_elements;};
	const size_t get_num_my_elements(){return num_my_elements;};
	const size_t get_first_my_global_index(){return first_my_global_index;};
	const size_t* get_element_size_list(){return element_size_list;};
	
protected:
	Comm& comm;
	std::array<size_t, dimension> tensor_total_size;
    std::array<size_t, dimension+1> tensor_total_size_mult;
    size_t num_global_elements = -1;
    size_t num_my_elements = -1;
    size_t first_my_global_index = -1;
    size_t* element_size_list;

};

}
