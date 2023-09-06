#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <cassert>
#include "mkl_wrapper.hpp"
#include "Tensor.hpp"

namespace TensorHetero{
template<typename datatype, size_t dimension>
class DenseTensor: public Tensor<datatype, dimension>{
public:
    std::vector<datatype> data;

    DenseTensor(){};
    DenseTensor(std::array<size_t, dimension> shape);
    DenseTensor(std::array<size_t, dimension> shape, std::vector<datatype> data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);
    //operator+
    //operator-
    //operator=

    void insert_value(std::array<size_t, dimension> index, datatype value);
    DenseTensor<datatype, dimension> clone() {return DenseTensor<datatype, dimension> (this->shape, this->data); };

};

template <typename datatype, size_t dimension>
DenseTensor<datatype, dimension>::DenseTensor(std::array<size_t, dimension> shape){
    this->shape = shape;
    //cumulative product
    this->shape_mult[0] = 1;
    for(size_t i = 0; i < dimension; ++i){
        this->shape_mult[i+1] = this->shape_mult[i] * shape[i];
    }
    //cumulative product end
    assert(this->shape_mult[dimension] != 0);
    this->data = std::vector<datatype>(this->shape_mult[dimension], 0);
}

template <typename datatype, size_t dimension>
DenseTensor<datatype, dimension>::DenseTensor(std::array<size_t, dimension> shape, std::vector<datatype> data){
    this->shape = shape;
    //cumulative product
    this->shape_mult[0] = 1;
    for(size_t i = 0; i < dimension; ++i){
        this->shape_mult[i+1] = this->shape_mult[i] * shape[i];
    }
    //cumulative product end
    assert(this->shape_mult[dimension] != 0);
    assert(this->shape_mult[dimension] == data.size() );
    this->data = data;
}

template <typename datatype, size_t dimension>
datatype &DenseTensor<datatype, dimension>::operator()(const std::array<size_t, dimension> index){
    // combined index : index[0] + index[1] * dims[0] + index[2] * dimes[0] * dims[1] + ...
    // i.e. 2D matrix : column major, i + row*j
    size_t combined_index = 0;
    for(size_t i = 0; i < dimension; ++i){
        combined_index += index[i] * this->shape_mult[i];
    }
    return this->data[combined_index];
}

template <typename datatype, size_t dimension>
datatype &DenseTensor<datatype, dimension>::operator[](size_t index){
    return this->data[index];
}
/*
template <typename datatype, size_t dimension>
DenseTensor<datatype, dimension> operator+(DenseTensor<datatype, dimension>& a, DenseTensor<datatype, dimension>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), 1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}

template <typename datatype, size_t dimension>
DenseTensor<datatype, dimension> operator-(DenseTensor<datatype, dimension>& a, DenseTensor<datatype, dimension>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), -1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}
*/

/*
template <class datatype, size_t dimension>
DenseTensor<datatype, dimension>& DenseTensor<datatype, dimension>::operator=(const DenseTensor<datatype, dimension>& tensor){
    this->shape = tensor.shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0 );
    this->data = tensor.data;

    return *this;
}
*/

template <typename datatype, size_t dimension>
void DenseTensor<datatype, dimension>::insert_value(std::array<size_t, dimension> index, datatype value){
    this->operator()(index) = value;
    return;
}


};
