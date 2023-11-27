#pragma once
#include <array>
#include "Comm.hpp"
#include "Map.hpp"
#include "decomposition/DecomposeResult.hpp"

namespace SE{

enum class STORETYPE{//data store type
    Dense,
    COO,
};

template<STORETYPE storetype, typename datatype, size_t dimension, typename computEnv, typename maptype>
class Tensor{
public:
    const STORETYPE store_type = storetype;
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    //std::vector<std::array<size_t, dimenion> > index;
    datatype* data;

    Comm<computEnv>* comm;
    maptype* map;

    Tensor(){};
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape): comm(_comm), shape(_shape){
        this->map = _map;
        cumprod<dimension>(this->shape, this->shape_mult);
        assert(this->shape_mult[dimension] != 0);
    };
    
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, datatype* _data): comm(_comm), shape(_shape), data(_data){
        this->map = _map;
        cumprod<dimension>(this->shape, this->shape_mult);
        assert(this->shape_mult[dimension] != 0);
    };


    datatype& operator()(const std::array<size_t, dimension> index){
        std::cout << "non-specified tensor." << std::endl;
        exit(1);
    };

    //bool get_filled() {return filled;};
    //void complete();
    //void insert_value(std::array<size_t, dimension> index, datatype value);
    //void print_tensor();
    Tensor<storetype, datatype, dimension, computEnv, maptype> clone(){
        return Tensor<storetype, datatype, dimension, computEnv, maptype> (this->comm, this->map, this->shape, this->data);
    }
};
}
