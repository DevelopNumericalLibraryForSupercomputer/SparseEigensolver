#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Comm.hpp"
#include "Utility.hpp"
namespace SE{

template<int dimension, MTYPE mtype>
class MapInp;

template<int dimension, MTYPE mtype>
class Map{
    using array_d =std::array<int, dimension>;
public:
    MTYPE _mtype = mtype;
    // length of all_local_shape is equal to the number of procs 
    //std::array<int, dimension+1> global_shape_mult;

    //Constructor 
    Map(){};
    Map(array_d global_shape, int my_rank, int world_size): global_shape(global_shape), my_rank(my_rank), world_size(world_size){};
    //Map(array_d global_shape, int my_rank, int world_size, std::array<bool, dimension> is_parallel );
    Map(array_d global_shape, int my_rank, int world_size, array_d ranks_per_dim ): global_shape(global_shape), my_rank(my_rank), world_size(world_size), ranks_per_dim(ranks_per_dim){};

    //clone
    virtual std::unique_ptr<Map<dimension, mtype>> clone() const=0;

    // get shape 
    array_d get_global_shape()const{return global_shape;};
    void set_local_shape(array_d global_shape) const{ this->global_shape=global_shape;};
    array_d get_local_shape() const{return local_shape;};
    int get_global_shape(const int dim)const { return global_shape[dim];};
    void set_local_shape(const int dim, const int val) const{ this->global_shape[dim]=val;};
    int get_local_shape(const int dim) const { return local_shape[dim];};

    // get number of elements
    int get_num_global_elements() const {return global_shape_mult[dimension]; };
    virtual int get_num_local_elements() const=0;

    // global index <-> local index
    virtual int  local_to_global(const int local_index) const=0;
    virtual int  global_to_local (const int global_index) const=0;
    virtual array_d local_to_global(const array_d local_array_index) const=0;
    virtual array_d global_to_local (const array_d global_array_index) const=0;

    // local array index <-> local index 
    virtual int  unpack_local_array_index(array_d local_array_index) const=0;
    virtual array_d pack_local_index(int local_index) const =0;
    virtual int  unpack_global_array_index(array_d global_array_index) const =0;
    virtual array_d pack_global_index(int global_index) const =0;

    virtual int find_rank_from_global_index(int global_index) const= 0;
    virtual int find_rank_from_global_array_index(array_d global_array_index) const= 0;

	virtual array_d get_nprow() const {array_d return_val; return_val.fill(1); return return_val;};
	virtual array_d get_block_size() const {array_d return_val; return_val.fill(1); return return_val;};

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

	virtual std::unique_ptr<MapInp<dimension, mtype> > generate_map_inp() const=0;

protected:
    array_d global_shape;
    array_d local_shape;
    int my_rank;
    int world_size;
    array_d ranks_per_dim;
    std::array<int, dimension+1> global_shape_mult;
    std::array<int, dimension+1> local_shape_mult;

    //std::array<std::vector<int>, dimension> partition_size_of_dim;
    //std::array<std::vector<int>, dimension> start_global_index_of_dim;
    //void initialize();
    /*
    int num_global_elements = -1;
    int num_my_elements = -1;
    int first_my_global_index = -1;
    int* element_size_list;
    */
    //int world_size;

    //bool is_sliced;
    //int sliced_dimension; // if is_sliced is false, sliced_dimension will become 0


};
template<int dimension, MTYPE mtype>
class MapInp
{
	public:
		//common
		std::array<int, dimension >	 global_shape;
		int my_rank;
		int world_size;

		//Contiguous1D
		std::array<int, dimension> ranks_per_dim;

		//BlockCycling
		std::array<int, dimension> block_size;
		std::array<int, dimension> nprow;

		// function
		virtual std::unique_ptr<Map<dimension,mtype> > create_map()=0; 
};

/*
template <int dimension>
Map<dimension>::Map(array_d total_size, int world_size){
    this->global_shape = total_size;
    cumprod(total_size, this->global_shape_mult);
    this->world_size = world_size;
    this->is_sliced = false;
    this->sliced_dimension = 0;
    
    this->partition_size = new int[1];
    this->start_global_index = new int[1];
    this->partition_size[0] = total_size[0];
    this->start_global_index[0] = 0;
    
}
template <int dimension>
Map<dimension>::Map(array_d total_size, int world_size, int slice_dimension){
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
//template <int dimension>
//void Map<dimension>::initialize(bool slice, int slice_dimension){}

}
