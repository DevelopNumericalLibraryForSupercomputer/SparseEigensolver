#pragma once
#include <cassert>
#include <algorithm>
#include <iomanip>
#include "Map.hpp"
namespace SE{
template<int dimension> 
class Contiguous1DMap: public Map<dimension,MTYPE::Contiguous1D> {
    using array_d = std::array<int, dimension>;

public:
    // constructor
    Contiguous1DMap(){};
    Contiguous1DMap( const array_d global_shape, const int my_rank, const int world_size);
    Contiguous1DMap( const array_d global_shape, const int my_rank, const int world_size, const std::array<bool, dimension> is_parallel );
    Contiguous1DMap( const array_d global_shape, const int my_rank, const int world_size, const array_d ranks_per_dim );

    Contiguous1DMap<dimension>* clone() const override{
        return new Contiguous1DMap(this->global_shape, this->my_rank, this->world_size, this->ranks_per_dim);
    };
    // get number of elements
    int get_num_local_elements() const override {return this->local_shape_mult[dimension]; };

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
    // all_local_shape store local_shape of all ranks, so all_local_shape.size() == world_size
    std::vector< array_d > all_local_shape;
    int split_dim=0;
    void initialize();
};

template <int dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( std::array<int, dimension> global_shape, int my_rank, int world_size)
:Map<dimension, MTYPE::Contiguous1D>(global_shape, my_rank, world_size){
    cumprod<dimension>(this->global_shape, this->global_shape_mult);
	
    this->ranks_per_dim.fill(1);
    this->ranks_per_dim[0]=this->world_size;

    // ranks_per_dim[0] = world_size --> dim 0 is paritioned by world_size;
    split_dim = 0;
    initialize();
};
template<int dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( std::array<int, dimension> global_shape, int my_rank, int world_size, std::array<bool, dimension> is_parallel) 
:Map<dimension, MTYPE::Contiguous1D>(global_shape, my_rank, world_size){
    cumprod<dimension>(this->global_shape, this->global_shape_mult);

    assert (1==std::count(is_parallel.begin(), is_parallel.end(), true));
    split_dim = std::distance(is_parallel.begin(), std::find(is_parallel.begin(), is_parallel.end(), true));
    split_dim = split_dim==dimension? 0: split_dim;
    this->ranks_per_dim.fill(1);
    this->ranks_per_dim[split_dim] = this->world_size;
    initialize();
}

template<int dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( const std::array<int, dimension> global_shape, const int my_rank, const int world_size, const std::array<int, dimension> ranks_per_dim )
: Map<dimension,MTYPE::Contiguous1D>(global_shape, my_rank, world_size, ranks_per_dim){

    split_dim = std::distance(this->ranks_per_dim.begin(), std::find_if(this->ranks_per_dim.begin(), this->ranks_per_dim.end(), [](int d){return (d!=1);}) );
    split_dim = split_dim==dimension? 0: split_dim;
    assert (this->ranks_per_dim[split_dim] == this->world_size);
    initialize();
}

template <int dimension>
void Contiguous1DMap<dimension>::initialize(){

    assert (split_dim<dimension);

    for (int i_rank=0; i_rank< this->world_size; i_rank++){
        auto tmp_local_shape = this->global_shape;
        tmp_local_shape[split_dim] =  this->global_shape[split_dim] / this->ranks_per_dim[split_dim];
        tmp_local_shape[split_dim] += this->global_shape[split_dim] % this->ranks_per_dim[split_dim] > i_rank ? 1 : 0; //{m+1,m+1,...,m+1,m,m,...,m}

        all_local_shape.push_back(tmp_local_shape);
    }
    this->local_shape = all_local_shape[this->my_rank];
    cumprod<dimension>(this->local_shape, this->local_shape_mult);
    /*
    std::cout << "Rank " << this->my_rank << " : ";
    for(int i=0;i<this->local_shape.size();i++){
        std::cout << this->local_shape[i] << ", " << this->local_shape_mult[i] << " || ";
    }
    std::cout << this->local_shape_mult[this->local_shape.size()] << std::endl;
    */
    return;
}

template <int dimension>
int Contiguous1DMap<dimension>::local_to_global(const int local_index) const
{
    std::array<int, dimension> local_array_index = pack_local_index(local_index); 
    std::array<int, dimension> global_array_index = local_to_global(local_array_index);
    return unpack_global_array_index(global_array_index);
}

template <int dimension>
int Contiguous1DMap<dimension>::global_to_local(const int global_index) const
{
    std::array<int, dimension> global_array_index = pack_global_index(global_index); 
    std::array<int, dimension> local_array_index = global_to_local(global_array_index);
    return unpack_local_array_index(local_array_index);
}

template <int dimension>
std::array<int, dimension> Contiguous1DMap<dimension>::local_to_global(const std::array<int, dimension> local_array_index) const
{
    std::array<int, dimension> global_array_index = local_array_index;
    for(int i_rank = 0;i_rank<this->my_rank;i_rank++){
        global_array_index[split_dim] += all_local_shape[i_rank][split_dim];
    }
    return global_array_index;
}
template <int dimension>
std::array<int, dimension> Contiguous1DMap<dimension>::global_to_local(const std::array<int, dimension> global_array_index) const
{
    std::array<int, dimension> local_array_index = global_array_index;
    for(int i_rank = 0;i_rank<this->my_rank;i_rank++){
        local_array_index[split_dim] -= all_local_shape[i_rank][split_dim];
    }

    assert ( local_array_index[split_dim] < this->local_shape[split_dim]) ;

    return local_array_index;
}

template <int dimension>
int  Contiguous1DMap<dimension>::unpack_local_array_index(std::array<int, dimension> local_array_index) const{
    int local_index = local_array_index[0];
    for(int i=1;i<dimension;i++){
        local_index *= this->local_shape[i];
        local_index += local_array_index[i];
    }
    return local_index;
}

template <int dimension>
std::array<int, dimension> Contiguous1DMap<dimension>::pack_local_index(int local_index) const{
    std::array<int, dimension>  local_array_index;
    int tmp_index = local_index;
    for(int i = dimension-1;i>0;i--){
        local_array_index[i] = tmp_index% this->local_shape[i];
        tmp_index /= this->local_shape[i];
    }
    local_array_index[0] = tmp_index;
    return local_array_index;
}

template <int dimension>
int  Contiguous1DMap<dimension>::unpack_global_array_index( std::array<int, dimension> global_array_index) const{
    int global_index = global_array_index[0];
    for(int i=1;i<dimension;i++){
        global_index *= this->global_shape[i];
        global_index += global_array_index[i];
    }
    return global_index;
}

template <int dimension>
std::array<int, dimension> Contiguous1DMap<dimension>::pack_global_index(int global_index) const{
    std::array<int, dimension> global_array_index;
    int tmp_index = global_index;
    for(int i = dimension-1;i>0;i--){
        global_array_index[i] = tmp_index% this->global_shape[i];
        tmp_index /= this->global_shape[i];
    }
    global_array_index[0] = tmp_index;
    return global_array_index;
}

template <int dimension>
int Contiguous1DMap<dimension>::find_rank_from_global_index(int global_index) const{
   return find_rank_from_global_array_index(pack_global_index(global_index)); 
}

template <int dimension>
int Contiguous1DMap<dimension>::find_rank_from_global_array_index(std::array<int, dimension> global_array_index) const{
    int temp_index = global_array_index[split_dim];

    for(int i_rank = 0; i_rank<this->world_size ; i_rank++){
        if(temp_index<all_local_shape[i_rank][split_dim]){
            return i_rank;
        }
        else{
            temp_index -= all_local_shape[i_rank][split_dim];
        }
    }
    assert(true);
    return 0;
}

}
