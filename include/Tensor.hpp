#pragma once
#include <array>
#include <iostream>
#include "Device.hpp"
#include "Comm.hpp"
namespace TensorHetero{

template<typename datatype, size_t dimension, typename device, typename comm>
class Tensor{
public:
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    virtual void complete(){};
    virtual bool get_filled() {return true;};
    virtual void insert_value(std::array<size_t, dimension> index, datatype value) = 0;
        
    //virtual datatype* unfold(Tensor<datatype,dimension,device,comm> tensor, size_t axis){};
    //virtual datatype* fold(Tensor<datatype,dimension,device,comm> tensor, size_t axis){};
};
}