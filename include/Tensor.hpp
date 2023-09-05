#pragma once
#include <array>
#include <stdio.h>

namespace TensorHetero{

template<typename datatype, size_t dimension>
class Tensor{
public:
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    virtual void complete(){};
    virtual bool get_filled() {return true;};
    virtual void insert_value(std::array<size_t, dimension> index, datatype value) = 0;
};
}