#pragma once
#include <array>
#include "Comm.hpp"
//#include "Comm_include.hpp"
#include "Map.hpp"
#include "decomposition/DecomposeResult.hpp"

namespace SE{

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
class Tensor{
public:
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    Comm<computEnv>* comm;
    maptype* map;

    Tensor(){};
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape): comm(_comm), map(_map), shape(_shape){
        cumprod<dimension>(this->shape, this->shape_mult);
        assert(this->shape_mult[dimension] != 0);
    };

    virtual datatype& operator()(const std::array<size_t, dimension> index){std::cout << "non-specified tensor." << std::endl; exit(1);};

    virtual void complete(){};
    virtual bool get_filled() {return true;};
    virtual void insert_value(std::array<size_t, dimension> index, datatype value){};
    virtual void print_tensor(){};
    //virtual Tensor<datatype, dimension, comm, map> clone() {};

protected:
    bool filled;
};
}
