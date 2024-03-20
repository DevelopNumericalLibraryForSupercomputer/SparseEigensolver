#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Comm.hpp"
#include "Utility.hpp"
namespace SE{


template<size_t dimension, MTYPE mtype>
class Map{
    using array_d =std::array<size_t, dimension>;
public:
    MTYPE _mtype = mtype;
    // length of all_local_shape is equal to the number of procs 
    //std::array<size_t, dimension+1> global_shape_mult;

    //Constructor 
    Map(){};
    Map(array_d global_shape, size_t my_rank, size_t world_size): global_shape(global_shape), my_rank(my_rank), world_size(world_size){};
    //Map(array_d global_shape, size_t my_rank, size_t world_size, std::array<bool, dimension> is_parallel );
    Map(array_d global_shape, size_t my_rank, size_t world_size, array_d ranks_per_dim ): global_shape(global_shape), my_rank(my_rank), world_size(world_size), ranks_per_dim(ranks_per_dim){};

    //clone
    virtual Map<dimension, mtype>* clone() const=0;

    // get shape 
    array_d get_global_shape()const{return global_shape;};
    array_d get_local_shape() const{return local_shape;};
    size_t get_global_shape(const size_t dim)const { return global_shape[dim];};
    size_t get_local_shape(const size_t dim) const { return local_shape[dim];};

    // get number of elements
    size_t get_num_global_elements() const {return global_shape_mult[dimension]; };
    virtual size_t get_num_local_elements() const=0;

    // global index <-> local index
    virtual size_t  local_to_global(const size_t local_index) const=0;
    virtual size_t  global_to_local (const size_t global_index) const=0;
    virtual array_d local_to_global(const array_d local_array_index) const=0;
    virtual array_d global_to_local (const array_d global_array_index) const=0;

    // local array index <-> local index 
    virtual size_t  unpack_local_array_index(array_d local_array_index) const=0;
    virtual array_d pack_local_index(size_t local_index) const =0;
    virtual size_t  unpack_global_array_index(array_d global_array_index) const =0;
    virtual array_d pack_global_index(size_t global_index) const =0;

    virtual size_t find_rank_from_global_index(size_t global_index) const= 0;
    virtual size_t find_rank_from_global_array_index(array_d global_array_index) const= 0;

    friend std::ostream& operator<< (std::ostream& stream, const Map<dimension,mtype>& map) {
        std::cout << "========= Map Info =========" <<std::endl;
        std::cout << "type: " << (int) map._mtype << "\n" 
                  << "shape: ("  ;
        for (auto shape_i : map.get_global_shape()){
            std::cout << shape_i << ",";
        }
        std::cout << ")" << std::endl;
        return stream;
    }

protected:
    array_d global_shape;
    array_d local_shape;
    size_t my_rank;
    size_t world_size;
    array_d ranks_per_dim;
    std::array<size_t, dimension+1> global_shape_mult;
    std::array<size_t, dimension+1> local_shape_mult;

    //std::array<std::vector<size_t>, dimension> partition_size_of_dim;
    //std::array<std::vector<size_t>, dimension> start_global_index_of_dim;
    //void initialize();
    /*
    size_t num_global_elements = -1;
    size_t num_my_elements = -1;
    size_t first_my_global_index = -1;
    size_t* element_size_list;
    */
    //size_t world_size;

    //bool is_sliced;
    //size_t sliced_dimension; // if is_sliced is false, sliced_dimension will become 0


};
/*
template <size_t dimension>
Map<dimension>::Map(array_d total_size, size_t world_size){
    this->global_shape = total_size;
    cumprod(total_size, this->global_shape_mult);
    this->world_size = world_size;
    this->is_sliced = false;
    this->sliced_dimension = 0;
    
    this->partition_size = new size_t[1];
    this->start_global_index = new size_t[1];
    this->partition_size[0] = total_size[0];
    this->start_global_index[0] = 0;
    
}
template <size_t dimension>
Map<dimension>::Map(array_d total_size, size_t world_size, size_t slice_dimension){
    assert(slice_dimension < total_size.size());
    this->global_shape = total_size;
    cumprod(total_size, this->global_shape_mult);
    this->world_size = world_size;
    if(this->world_size >1){
        initialize(true, slice_dimension);
    }
    else{
        this->is_sliced = false;
        this->sliced_dimension = 0;
    }
}
*/
//template <size_t dimension>
//void Map<dimension>::initialize(bool slice, size_t slice_dimension){}

}
