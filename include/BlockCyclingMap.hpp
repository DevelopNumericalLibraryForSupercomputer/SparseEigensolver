#pragma once
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "Map.hpp"
namespace SE{
template<int dimension> 

/*
Map for Block Cyclic Distribution (for all dimension) 
nprow: number of processor for each row
block_size: size of block 
world_size == nprow[0] * nprow[1] .... 

block index: 

*/
class BlockCyclingMap: public Map<dimension,MTYPE::BlockCycling> {
    using array_d = std::array<int, dimension>;

public:
    // constructor
    BlockCyclingMap(){};
    BlockCyclingMap( const array_d global_shape, const int my_rank, const int world_size, const array_d block_size, const array_d nprow);

    BlockCyclingMap<dimension>* clone() const override{
        return new BlockCyclingMap(this->global_shape, this->my_rank, this->world_size, this->block_size, this->nprow);
    };
    // get number of elements
    int get_num_local_elements() const override;

    // global index <-> local index
    int local_to_global(const int local_index)  const override;
    int global_to_local(const int global_index) const override;
    array_d local_to_global(const array_d local_array_index) const override;
    array_d global_to_local(const array_d global_array_index) const override;
                                                                                
    // local array index <-> local index 
    int  unpack_local_array_index(array_d local_array_index) const override;
    array_d pack_local_index(int local_index) const override;
    int  unpack_global_array_index(array_d global_array_index) const override;
    array_d pack_global_index(int global_index) const override;

    int find_rank_from_global_index(int global_index) const override;
    int find_rank_from_global_array_index(array_d global_array_index) const override;

    std::vector< array_d > get_all_local_shape() const{return all_local_shape;};
    int get_split_dim() const {return split_dim;};
private:
    array_d nprow;
	array_d block_size;
};

template <int dimension>
BlockCyclingMap<dimension>::BlockCyclingMap( const array_d global_shape, const int my_rank, const int world_size, const array_d block_size, 
											 const array_d nprow)
:Map<dimension, MTYPE::BlockCycling>(global_shape, my_rank, world_size){
    assert( world_size == std::accumulate(nprow.begin(), nprow.end(), 1, std::multiplies<int>()) );

	this->block_size=block_size;
	this->nprow = nprow;
	
}
template <int dimension>
int BlockCyclingMap<dimension>::get_num_local_elements() const 
{

	int num_elements =1;
	for (int dim = 0; dim<dimension; dim++){
		int num_elements_per_dim = 0;
		for (int i = this->my_rank; i<std::ceil((double)this->global_shape[dim] / (double)block_size[dim]); i+=nprow[dim] ){
			int start_idx = i*block_size[dim];
			int end_idx = std::min((i+1)*block_size[dim], this->global_shape[dim] );
			num_elements_per_dim += end_idx-start_idx;
		}
		num_elements *=num_elements_per_dim;
	}
	return num_elements;	
}

template <int dimension>
int BlockCyclingMap<dimension>::local_to_global(const int local_index) const
{
    std::array<int, dimension> local_array_index = pack_local_index(local_index); 
    std::array<int, dimension> global_array_index = local_to_global(local_array_index);
    return unpack_global_array_index(global_array_index);
}

template <int dimension>
int BlockCyclingMap<dimension>::global_to_local(const int global_index) const
{
    std::array<int, dimension> global_array_index = pack_global_index(global_index); 
//	printf("%d", global_index );
//	for (int i =0; i<dimension; i++){
//		printf("  %d", global_array_index[i]);	
//	}
//	printf("\n");
    std::array<int, dimension> local_array_index = global_to_local(global_array_index);
    return unpack_local_array_index(local_array_index);
}

template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::local_to_global(const std::array<int, dimension> local_array_index) const
{
    std::array<int, dimension> global_array_index;

    // Calculate global block index for each dimension and calculate global index
    for (int dim = 0; dim < local_array_index.size(); ++dim) {
		//local_index[dim] / this->block_size[dim] is local block index
		int global_block_index = this->nprow[dim] * (local_array_index[dim] / this->block_size[dim])+this->my_rank;
        global_array_index[dim] = global_block_index * this->block_size[dim] + local_array_index[dim] % this->block_size[dim];
    }

    return global_array_index;
}

template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::global_to_local(const std::array<int, dimension> global_array_index) const
{
    std::array<int, dimension> local_array_index;
    std::array<int, dimension> global_block_index;
	bool tag = true;
    for (int dim = 0; dim < local_array_index.size(); ++dim) {
		global_block_index[dim] = global_array_index[dim] / this->block_size[dim];
		local_array_index[dim] = (global_block_index[dim]/this->nprow[dim])*this->block_size[dim] + global_array_index[dim]%this->block_size[dim];

		if(this->my_rank!=(global_block_index[dim]%this->nprow[dim]) ){
			tag = false;
			break;
		}
	}
	if(!tag){
		for (int dim = 0; dim < local_array_index.size(); ++dim) {
			local_array_index[dim] = -1;
		}
	}

	return 	local_array_index;
}







template <int dimension>
int BlockCyclingMap<dimension>::unpack_local_array_index(std::array<int, dimension> local_array_index) const {
    int local_index = local_array_index[dimension - 1];
    for (int i = dimension - 2; i >= 0; i--) {
        local_index *= this->local_shape[i];
        local_index += local_array_index[i];
    }
    return local_index;
}

template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::pack_local_index(int local_index) const {
    std::array<int, dimension> local_array_index;
    int tmp_index = local_index;
    for (int i = 0; i < dimension - 1; i++) {
        local_array_index[i] = tmp_index % this->local_shape[i];
        tmp_index /= this->local_shape[i];
    }
    local_array_index[dimension - 1] = tmp_index;
    return local_array_index;
}
/*
template <int dimension>
int  BlockCyclingMap<dimension>::unpack_local_array_index(std::array<int, dimension> local_array_index) const{
    int local_index = local_array_index[0];
    for(int i=1;i<dimension;i++){
        local_index *= this->local_shape[i];
        local_index += local_array_index[i];
    }
    return local_index;
}

template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::pack_local_index(int local_index) const{
    std::array<int, dimension>  local_array_index;
    int tmp_index = local_index;
    for(int i = dimension-1;i>0;i--){
        local_array_index[i] = tmp_index% this->local_shape[i];
        tmp_index /= this->local_shape[i];
    }
    local_array_index[0] = tmp_index;
    return local_array_index;
}
*/
/*
template <int dimension>
int  BlockCyclingMap<dimension>::unpack_global_array_index( std::array<int, dimension> global_array_index) const{
    int global_index = global_array_index[0];
    for(int i=1;i<dimension;i++){
        global_index *= this->global_shape[i];
        global_index += global_array_index[i];
    }
    return global_index;
}*/
template <int dimension>
int BlockCyclingMap<dimension>::unpack_global_array_index(std::array<int, dimension> global_array_index) const {
    int global_index = global_array_index[dimension - 1];
    for (int i = dimension - 2; i >= 0; i--) {
        global_index *= this->global_shape[i];
        global_index += global_array_index[i];
    }
    return global_index;
}

/*
template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::pack_global_index(int global_index) const{
    std::array<int, dimension> global_array_index;
    int tmp_index = global_index;
    for(int i = dimension-1;i>0;i--){
        global_array_index[i] = tmp_index% this->global_shape[i];
        tmp_index /= this->global_shape[i];
    }
    global_array_index[0] = tmp_index;
    return global_array_index;
}*/

template <int dimension>
std::array<int, dimension> BlockCyclingMap<dimension>::pack_global_index(int global_index) const{
    std::array<int, dimension> global_array_index;
    int tmp_index = global_index;
    for(int i = 0; i < dimension - 1; i++){
        global_array_index[i] = tmp_index % this->global_shape[i];
        tmp_index /= this->global_shape[i];
    }
    global_array_index[dimension - 1] = tmp_index;
    return global_array_index;
}


template <int dimension>
int BlockCyclingMap<dimension>::find_rank_from_global_index(int global_index) const{
   return find_rank_from_global_array_index(pack_global_index(global_index)); 
}

template <int dimension>
int BlockCyclingMap<dimension>::find_rank_from_global_array_index(std::array<int, dimension> global_array_index) const{

    int proc_id = 0;
    int stride = 1;

    // Iterate over each dimension of the tensor
    for (int dim = global_array_index.size() - 1; dim >= 0; --dim) {
        proc_id += (global_array_index[dim] / this->block_size[dim] % nprow[dim]) * stride;
        stride *= nprow[dim];
    }

    return proc_id;	
}

}
