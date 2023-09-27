#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Tensor.hpp"
#include "Device.hpp"
#include "ContiguousMap.hpp"
#include "Utility_include.hpp"
#include "DecomposeResult.hpp"
namespace SE{
template<typename datatype, size_t dimension, Comm comm, Map map>{


}
template<typename datatype, size_t dimension, typename device, typename map>
class DenseTensor: public Tensor<datatype, dimension, device, map>{
public:
    datatype* data;

    DenseTensor(){};
    DenseTensor(std::array<size_t, dimension> shape);
    DenseTensor(std::array<size_t, dimension> shape, datatype* data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);
    //operator+
    //operator-
    DenseTensor<datatype, dimension, device, map>& operator=(const DenseTensor<datatype, dimension, device, map> &tensor);

    //void complete(){};
    //bool get_filled() {return true;};
    void insert_value(std::array<size_t, dimension> index, datatype value);
    DenseTensor<datatype, dimension, device, map> clone() {return DenseTensor<datatype, dimension, device, map> (this->shape, this->data); };
    DecomposeResult<datatype, dimension, device> decompose(std::string method);

};

template <typename datatype, size_t dimension, typename device, typename map>
DenseTensor<datatype, dimension, device, map>::DenseTensor(std::array<size_t, dimension> shape){
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    //this->data = new datatype[this->shape_mult[dimension]];
    this->data = malloc<datatype, device>(this->shape_mult[dimension]);
}

template <typename datatype, size_t dimension, typename device, typename map>
DenseTensor<datatype, dimension, device, map>::DenseTensor(std::array<size_t, dimension> shape, datatype* data){
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    //assert(this->shape_mult[dimension] == data.size() ); We don't know.
    this->data = data;
}

template <typename datatype, size_t dimension, typename device, typename map>
datatype &DenseTensor<datatype, dimension, device, map>::operator()(const std::array<size_t, dimension> index){
    // combined index : index[0] + index[1] * dims[0] + index[2] * dimes[0] * dims[1] + ...
    // i.e. 2D matrix : column major, i + row*j
    size_t combined_index = 0;
    for(size_t i = 0; i < dimension; ++i){
        combined_index += index[i] * this->shape_mult[i];
    }
    return this->data[combined_index];
}

template <typename datatype, size_t dimension, typename device, typename map>
datatype &DenseTensor<datatype, dimension, device, map>::operator[](size_t index){
    return this->data[index];
}
/*
template <typename datatype, size_t dimension, typename device, typename map>
DenseTensor<datatype, dimension, device, map> operator+(DenseTensor<datatype, dimension, device, map>& a, DenseTensor<datatype, dimension, device, map>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), 1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}

template <typename datatype, size_t dimension, typename device, typename map>
DenseTensor<datatype, dimension> operator-(DenseTensor<datatype, dimension, device, map>& a, DenseTensor<datatype, dimension, device, map>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), -1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}
*/

template <class datatype, size_t dimension, typename device, typename map>
DenseTensor<datatype, dimension, device, map>& DenseTensor<datatype, dimension, device, map>::operator=(const DenseTensor<datatype, dimension, device, map>& tensor){
    if(this == &tensor){
        return *this;
    }
    this->shape = tensor.shape;
    this->shape_mult = tensor.shape_mult;
    // cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0 );

    //delete[] this->data;
    //this->data = new datatype[this->shape_mult[dimension]];
    free<datatype, device>(this->data);
    this->data = malloc<datatype, device>(this->shape_mult[dimension]);
    memcpy<datatype, device>(this->data, tensor.data, this->shape_mult[dimension]);
    return *this;
}


template <typename datatype, size_t dimension, typename device, typename map>
void DenseTensor<datatype, dimension, device, map>::insert_value(std::array<size_t, dimension> index, datatype value){
    this->operator()(index) = value;
    return;
}

template <typename datatype, size_t dimension, typename device, typename map>
DecomposeResult<datatype, dimension, device> DenseTensor<datatype, dimension, device, map>::decompose(std::string method){
    std::cout << method << " is not implemented yet." << std::endl;
    exit(-1);
}

};
