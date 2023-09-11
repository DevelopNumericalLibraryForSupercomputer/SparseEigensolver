#pragma once
#include <iostream>
#include <array>
#include "Comm.hpp"

namespace TensorHetero{
template<size_t dimension>
class Map{
public:
	Map(Comm *commPtr): comm(commPtr){};
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

	Comm* get_comm(){return comm;};
	
protected:
	Comm* comm;
};

}