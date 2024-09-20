#pragma once
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include "Map.hpp"
namespace SE{

template<int dimension>
class BlockCyclingMapInp;
/*
Map for Block Cyclic Distribution (for all dimension) 
nprow: number of processor for each row
block_size: size of block 
world_size == nprow[0] * nprow[1] .... 

array rank index is col-wise; which means "0->(0,0)" & "1->(1,0)" & "2->(0,1)" & "3->(1,1)"

block index: 

        0     1     0     1    0             (0,0)              (0,1)
      __________________________         ______________      ___________
     |1  1 |2  2 |3  3 |4  4 |5 |       |1  1 |3  3 |5 |    |2  2 |4  4 |
 0   |1  1 |2  2 |3  3 |4  4 |5 |       |1  1 |3  3 |5 |    |2  2 |4  4 |
     |_____|_____|_____|_____|__|       |_____|_____|__|    |_____|_____|
     |6  6 |7  7 |8  8 |9  9 |10|       |11 11|13 13|15|    |12 12|14 14|
 1   |6  6 |7  7 |8  8 |9  9 |10|       |11 11|13 13|15|    |12 12|14 14|
     |_____|_____|_____|_____|__|       |_____|_____|__|    |_____|_____|
     |11 11|12 12|13 13|14 14|15| --->  |21 21|23 23|25|    |22 22|24 24|
 0   |11 11|12 12|13 13|14 14|15|       |_____|_____|__|    |_____|_____|
     |_____|_____|_____|_____|__|
     |16 16|17 17|18 18|19 19|20|        
 1   |16 16|17 17|18 18|19 19|20|            (1,0)              (1,1)
     |_____|_____|_____|_____|__|        ______________      ___________
 0   |21 21|22 22|23 23|24 24|25|       |6  6 |8  8 |10|    |7  7 |9  9 |
     |_____|_____|_____|_____|__|       |6  6 |8  8 |10|    |7  7 |9  9 |
                                        |_____|_____|__|    |_____|_____|
                                        |16 16|18 18|20|    |17 17|19 19|
                                        |16 16|18 18|20|    |17 17|19 19|
                                        |_____|_____|__|    |_____|_____|

*/
template<int dimension> 
class BlockCyclingMap: public Map<dimension,MTYPE::BlockCycling> {
    using array_d = std::array<int, dimension>;

public:
    // constructor
    BlockCyclingMap(){};
    BlockCyclingMap( const array_d global_shape, const int my_rank, const int world_size, const array_d block_size, const array_d nprow);

    std::unique_ptr<Map<dimension,MTYPE::BlockCycling>> clone() const override{
        return std::make_unique<BlockCyclingMap<dimension>>(this->global_shape, this->my_rank, this->world_size, this->block_size, this->nprow);
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

    //std::vector< array_d > get_all_local_shape() const{return all_local_shape;};
    array_d get_nprow() const override{return this->nprow;};
    array_d get_my_array_rank() const {return this->my_array_rank;};
    array_d get_block_size() const override {return this->block_size;};
    std::unique_ptr<MapInp<dimension, MTYPE::BlockCycling> > generate_map_inp() const override{
        return  std::make_unique<BlockCyclingMapInp<dimension> > (this->global_shape, this->my_rank, this->world_size,this->block_size, this->nprow);
    };
private:
    array_d nprow;      // number of processor for each dimension
    array_d my_array_rank; // processor_id for each dimension
    array_d block_size; // block size for each dimension
};

template<int dimension>
class BlockCyclingMapInp: public MapInp<dimension, MTYPE::BlockCycling >{
    public:

        std::unique_ptr< Map<dimension,MTYPE::BlockCycling> > create_map() override;

        BlockCyclingMapInp( std::array<int, dimension> global_shape, int my_rank, int world_size, std::array<int, dimension> block_size, std::array<int, dimension> nprow){
            this->global_shape=global_shape;
            this->my_rank = my_rank;
            this->world_size = world_size;
            this->block_size = block_size;
            this->nprow=nprow;
        };

};

template<int dimension>
std::unique_ptr< Map<dimension,MTYPE::BlockCycling> > BlockCyclingMapInp<dimension>::create_map(){
    return std::make_unique<BlockCyclingMap<dimension> > ( this->global_shape, this->my_rank, this->world_size, this->block_size, this->nprow);
};

template <int dimension>
BlockCyclingMap<dimension>::BlockCyclingMap( const std::array<int,dimension> global_shape, const int my_rank, const int world_size, const std::array<int, dimension> block_size, 
                                             const std::array<int, dimension> nprow)
:Map<dimension, MTYPE::BlockCycling>(global_shape, my_rank, world_size){
    assert( world_size == std::accumulate(nprow.begin(), nprow.end(), 1, std::multiplies<int>()) );
    int idx = my_rank;
    for (int dim=0; dim<dimension; dim++){
        this->my_array_rank[dim] = idx % nprow[dim];
        idx  /= nprow[dim];
    }
    this->block_size=block_size;
    this->nprow = nprow;


    for (int dim = 0; dim<dimension; dim++){
        this->local_shape[dim]=0;
        for (int i = this->my_array_rank[dim]; i<std::ceil((double)this->global_shape[dim] / (double)block_size[dim]); i+=nprow[dim] ){
            int start_idx = i*block_size[dim];
            int end_idx = std::min((i+1)*block_size[dim], this->global_shape[dim] );
            this->local_shape[dim]+= end_idx-start_idx;
        }
    }       
    
}
template <int dimension>
int BlockCyclingMap<dimension>::get_num_local_elements() const 
{

    int num_elements =1;
    for (int dim = 0; dim<dimension; dim++){
        num_elements*=this->local_shape[dim];
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
        int global_block_index = this->nprow[dim] * (local_array_index[dim] / this->block_size[dim])+this->my_array_rank[dim];
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
    for (int dim = 0; dim < dimension; ++dim) {
        global_block_index[dim] =  global_array_index[dim] / this->block_size[dim];
        local_array_index[dim]  = (global_block_index[dim] / this->nprow[dim])*this->block_size[dim] + global_array_index[dim]%this->block_size[dim];

        if(this->my_array_rank[dim]!=(global_block_index[dim]%this->nprow[dim]) ){
            tag = false;
            break;
        }
    }
    if(!tag){
        for (int dim = 0; dim < local_array_index.size(); ++dim) {
            local_array_index[dim] = -1;
        }
    }
    return  local_array_index;
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
