#pragma once
#include <array>
#include "Comm.hpp"
#include "Map.hpp"
#include "ContiguousMap.hpp"
//#include "decomposition/DecomposeResult.hpp"

namespace SE{

enum class STORETYPE{//data store type
    Dense,
    COO,
};

template<STORETYPE storetype, typename datatype, size_t dimension, typename computEnv, typename maptype=ContiguousMap<dimension> >
class Tensor{
public:
    const STORETYPE store_type = storetype;
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    //std::vector<std::array<size_t, dimenion> > index;
    datatype* data;

    const Comm<computEnv>* comm;
    //const maptype* map;
    const maptype map;

    Tensor(){};
    
    Tensor(const Comm<computEnv>* _comm, const std::array<size_t, dimension> _shape, const bool is_sliced = false, const size_t sliced_dimension = 0)
    : comm(_comm), shape(_shape), map( is_sliced ? maptype(_shape, _comm->world_size, sliced_dimension) : maptype(_shape, _comm->world_size) ){
        cumprod<dimension>(this->shape, this->shape_mult);
        assert(this->shape_mult[dimension] != 0);
    };
    
    Tensor(const Comm<computEnv>* _comm, const std::array<size_t, dimension>& _shape, const datatype* _data, const bool is_sliced = false, const size_t sliced_dimension = 0)
    :Tensor(_comm, _shape, is_sliced, sliced_dimension){};
/*
    Tensor(const Comm<computEnv>* _comm, const std::array<size_t, dimension> _shape, const bool is_sliced = false, const size_t sliced_dimension = 0)
    : comm(_comm), shape(_shape), map( is_sliced ? maptype(_shape, _comm->world_size, sliced_dimension) : maptype(_shape, _comm->world_size) ){
        cumprod<dimension>(this->shape, this->shape_mult);
        assert(this->shape_mult[dimension] != 0);

        size_t datasize = this->shpae_mult[dimension];
        if(is_sliced){
            datasize = this->map->get_my_partitioned_data_size(comm->rank);
        }
        this->data = malloc<datatype, computEnv>(datasize);
        memset<datatype, computEnv>(this->data, 0.0, datasize);
    };
    
    Tensor(const Comm<computEnv>* _comm, const std::array<size_t, dimension>& _shape, const datatype* _data, const bool is_sliced = false, const size_t sliced_dimension = 0)
    :Tensor(_comm, _shape, is_sliced, sliced_dimension){
        if(this->map->is_sliced){
            memcpy<datatype, computEnv>( this->data, _data, this->map->get_my_partitioned_data_size(comm->rank));
        }
        else{
            memcpy<datatype, computEnv>( this->data, _data, this->shape_mult[dimension]);
        }
    };
*/
    datatype& operator()(const std::array<size_t, dimension> index){
        std::cout << "non-specified tensor." << std::endl;
        exit(1);
    };

    //bool get_filled() {return filled;};
    //void complete();
    //void insert_value(std::array<size_t, dimension> index, datatype value);
    //void print_tensor();
    Tensor<storetype, datatype, dimension, computEnv, maptype>* clone(){
        return new Tensor<storetype, datatype, dimension, computEnv, maptype> (this->comm, this->shape, this->data, this->map->is_sliced, this->map->sliced_dimension);
    }
};
}
