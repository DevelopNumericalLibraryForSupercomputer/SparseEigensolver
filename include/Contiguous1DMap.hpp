#pragma once
#include <cassert>
#include <algorithm>
#include "Map.hpp"
//#include "Gather.hpp"
namespace SE{
template<size_t dimension> 
class Contiguous1DMap: public Map<dimension,MTYPE::Contiguous1D> {
    //friend Gather<Contiguous1DMap>;
    using array_d = std::array<size_t, dimension>;

public:
    // constructor
    Contiguous1DMap( const array_d global_shape, const size_t my_rank=0, const size_t world_size=1);
    Contiguous1DMap( const array_d global_shape, const size_t my_rank, const size_t world_size, const std::array<bool, dimension> is_parallel );
    Contiguous1DMap( const array_d global_shape, const size_t my_rank, const size_t world_size, const array_d ranks_per_dim );

    Contiguous1DMap<dimension>* clone() const override{
        return new Contiguous1DMap(this->global_shape, this->my_rank, this->world_size, this->ranks_per_dim);
    };
    // get number of elements
    size_t get_num_local_elements() const override {return this->local_shape_mult[dimension]; };

    // global index <-> local index
    size_t local_to_global(const size_t local_index)  const override;
    size_t global_to_local(const size_t global_index) const override;
    array_d local_to_global(const array_d local_array_index) const override;
    array_d global_to_local(const array_d global_array_index) const override;
                                                                                
    // local array index <-> local index 
    size_t  unpack_local_array_index(array_d local_array_index) const override;
    array_d pack_local_index(size_t local_index) const override;
    size_t  unpack_global_array_index(array_d global_array_index) const override;
    array_d pack_global_index(size_t global_index) const override;

    size_t find_rank_from_global_index(size_t global_index) const override;
    size_t find_rank_from_global_array_index(array_d global_array_index) const override;
private:
    size_t split_dim=0;
    void initialize();
    std::vector< array_d > all_local_shape;
};

template <size_t dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( std::array<size_t, dimension> global_shape, size_t my_rank, size_t world_size)
:Map<dimension, MTYPE::Contiguous1D>(global_shape, my_rank, world_size){
    cumprod(this->global_shape, this->global_shape_mult);

    this->ranks_per_dim.fill(1);
    this->ranks_per_dim[0]=this->world_size;

    // ranks_per_dim[0] = world_size --> dim 0 is paritioned by world_size;
    split_dim = 0;
    initialize();
};
template<size_t dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( std::array<size_t, dimension> global_shape, size_t my_rank, size_t world_size, std::array<bool, dimension> is_parallel) 
:Map<dimension, MTYPE::Contiguous1D>(global_shape, my_rank, world_size){
    cumprod(this->global_shape, this->global_shape_mult);

    assert (1==std::count(is_parallel.begin(), is_parallel.end(), true));
    split_dim = std::distance(is_parallel.begin(), std::find(is_parallel.begin(), is_parallel.end(), true));
    split_dim = split_dim==dimension? 0: split_dim;
    this->ranks_per_dim.fill(1);
    this->ranks_per_dim[split_dim] = this->world_size;
    initialize();
}

template<size_t dimension>
Contiguous1DMap<dimension>::Contiguous1DMap( const std::array<size_t, dimension> global_shape, const size_t my_rank, const size_t world_size, const std::array<size_t, dimension> ranks_per_dim )
: Map<dimension,MTYPE::Contiguous1D>(global_shape, my_rank, world_size, ranks_per_dim){

    split_dim = std::distance(this->ranks_per_dim.begin(), std::find_if(this->ranks_per_dim.begin(), this->ranks_per_dim.end(), [](size_t d){return (d!=1);}) );
    split_dim = split_dim==dimension? 0: split_dim;
    assert (this->ranks_per_dim[split_dim] == this->world_size);
    initialize();
}

template <size_t dimension>
void Contiguous1DMap<dimension>::initialize(){

    assert (split_dim<dimension);

    for (size_t i_rank=0; i_rank< this->world_size; i_rank++){
        auto tmp_local_shape = this->global_shape;
        tmp_local_shape[split_dim] = tmp_local_shape[split_dim] / this->ranks_per_dim[split_dim];
        tmp_local_shape[split_dim] += tmp_local_shape[split_dim] % this->ranks_per_dim[split_dim] > i_rank ? 1 : 0;

        all_local_shape.push_back(tmp_local_shape);
    }
    this->local_shape = all_local_shape[this->my_rank];
    cumprod(this->local_shape, this->local_shape_mult);
    return;
}

template <size_t dimension>
size_t Contiguous1DMap<dimension>::local_to_global(const size_t local_index) const
{
    std::array<size_t, dimension> local_array_index = pack_local_index(local_index); 
    std::array<size_t, dimension> global_array_index = local_to_global(local_array_index);
    return unpack_global_array_index(global_array_index);
}

template <size_t dimension>
size_t Contiguous1DMap<dimension>::global_to_local(const size_t global_index) const
{
    std::array<size_t, dimension> global_array_index = pack_global_index(global_index); 
    std::array<size_t, dimension> local_array_index = global_to_local(global_array_index);
    return unpack_local_array_index(local_array_index);
}

template <size_t dimension>
std::array<size_t, dimension> Contiguous1DMap<dimension>::local_to_global(const std::array<size_t, dimension> local_array_index) const
{
    std::array<size_t, dimension> global_array_index = local_array_index;
    for(size_t i_rank = 0;i_rank<this->my_rank;i_rank++){
        global_array_index[split_dim] += all_local_shape[i_rank][split_dim];
    }
    return global_array_index;
}
template <size_t dimension>
std::array<size_t, dimension> Contiguous1DMap<dimension>::global_to_local(const std::array<size_t, dimension> global_array_index) const
{
    std::array<size_t, dimension> local_array_index = global_array_index;
    for(size_t i_rank = 0;i_rank<this->my_rank;i_rank++){
        local_array_index[split_dim] -= all_local_shape[i_rank][split_dim];
    }

    assert ( local_array_index[split_dim] < this->local_shape[split_dim]) ;

    return local_array_index;
}

template <size_t dimension>
size_t  Contiguous1DMap<dimension>::unpack_local_array_index(std::array<size_t, dimension> local_array_index) const{
    size_t local_index = local_array_index[0];
    for(size_t i=1;i<dimension;i++){
        local_index *= this->local_shape[i];
        local_index += local_array_index[i];
    }
    return local_index;
}

template <size_t dimension>
std::array<size_t, dimension> Contiguous1DMap<dimension>::pack_local_index(size_t local_index) const{
    std::array<size_t, dimension>  local_array_index;
    size_t tmp_index = local_index;
    for(size_t i = dimension-1;i>0;i--){
        local_array_index[i] = tmp_index% this->local_shape[i];
        tmp_index /= this->local_shape[i];
    }
    local_array_index[0] = tmp_index;
    return local_array_index;
}

template <size_t dimension>
size_t  Contiguous1DMap<dimension>::unpack_global_array_index( std::array<size_t, dimension> global_array_index) const{
    size_t global_index = global_array_index[0];
    for(size_t i=1;i<dimension;i++){
        global_index *= this->global_shape[i];
        global_index += global_array_index[i];
    }
    return global_index;
}

template <size_t dimension>
std::array<size_t, dimension> Contiguous1DMap<dimension>::pack_global_index(size_t global_index) const{
    std::array<size_t, dimension> global_array_index;
    size_t tmp_index = global_index;
    for(size_t i = dimension-1;i>0;i--){
        global_array_index[i] = tmp_index% this->global_shape[i];
        tmp_index /= this->global_shape[i];
    }
    global_array_index[0] = tmp_index;
    return global_array_index;
}

template <size_t dimension>
size_t Contiguous1DMap<dimension>::find_rank_from_global_index(size_t global_index) const{
   return find_rank_from_global_array_index(pack_global_index(global_index)); 
}

template <size_t dimension>
size_t Contiguous1DMap<dimension>::find_rank_from_global_array_index(std::array<size_t, dimension> global_array_index) const{
    size_t temp_index = global_array_index[split_dim];

    for(size_t i_rank = 0; i_rank<this->world_size ; i_rank++){
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
