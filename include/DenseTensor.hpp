#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Tensor.hpp"

#include "decomposition/Utility.hpp"
namespace SE{
template<typename datatype, size_t dimension, typename comm, typename map>
class DenseTensor: public Tensor<datatype, dimension, comm, map>{
public:
    datatype* data;

    DenseTensor(){};
    DenseTensor(std::array<size_t, dimension> shape);
    DenseTensor(std::array<size_t, dimension> shape, datatype* data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);
    //operator+
    //operator-
    DenseTensor<datatype, dimension, comm, map>& operator=(const DenseTensor<datatype, dimension, comm, map> &tensor);

    //void complete(){};
    //bool get_filled() {return true;};
    void insert_value(std::array<size_t, dimension> index, datatype value);
    DenseTensor<datatype, dimension, comm, map> clone() {return DenseTensor<datatype, dimension, comm, map> (this->shape, this->data); };
    std::unique_ptr<DecomposeResult<datatype, dimension, comm, map> > decompose(const std::string method);

    //TEMPORAL
    std::unique_ptr<DecomposeResult<datatype, dimension, comm, map> > davidson(const std::string method);

};

template <typename datatype, size_t dimension, typename comm, typename map>
DenseTensor<datatype, dimension, comm, map>::DenseTensor(std::array<size_t, dimension> shape){
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    //this->data = new datatype[this->shape_mult[dimension]];
    this->data = malloc<datatype, comm::env>(this->shape_mult[dimension]);
}

template <typename datatype, size_t dimension, typename comm, typename map>
DenseTensor<datatype, dimension, comm, map>::DenseTensor(std::array<size_t, dimension> shape, datatype* data){
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    //assert(this->shape_mult[dimension] == data.size() ); We don't know.
    this->data = malloc<datatype, comm::env>(this->shape_mult[dimension]);
    memcpy<datatype,comm::env>( this->data, data, this->shape_mult[dimension]);
}

template <typename datatype, size_t dimension, typename comm, typename map>
datatype &DenseTensor<datatype, dimension, comm, map>::operator()(const std::array<size_t, dimension> index){
    // combined index : index[0] + index[1] * dims[0] + index[2] * dimes[0] * dims[1] + ...
    // i.e. 2D matrix : column major, i + row*j
    size_t combined_index = 0;
    for(size_t i = 0; i < dimension; ++i){
        combined_index += index[i] * this->shape_mult[i];
    }
    return this->data[combined_index];
}

template <typename datatype, size_t dimension, typename comm, typename map>
datatype &DenseTensor<datatype, dimension, comm, map>::operator[](size_t index){
    return this->data[index];
}
/*
template <typename datatype, size_t dimension, typename comm, typename map>
DenseTensor<datatype, dimension, comm, map> operator+(DenseTensor<datatype, dimension, comm, map>& a, DenseTensor<datatype, dimension, comm, map>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), 1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}

template <typename datatype, size_t dimension, typename comm, typename map>
DenseTensor<datatype, dimension> operator-(DenseTensor<datatype, dimension, comm, map>& a, DenseTensor<datatype, dimension, comm, map>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), -1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}
*/

template <class datatype, size_t dimension, typename comm, typename map>
DenseTensor<datatype, dimension, comm, map>& DenseTensor<datatype, dimension, comm, map>::operator=(const DenseTensor<datatype, dimension, comm, map>& tensor){
    if(this == &tensor){
        return *this;
    }
    this->shape = tensor.shape;
    this->shape_mult = tensor.shape_mult;
    // cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0 );

    //delete[] this->data;
    //this->data = new datatype[this->shape_mult[dimension]];
    free<datatype, comm>(this->data);
    this->data = malloc<datatype, comm>(this->shape_mult[dimension]);
    memcpy<datatype, comm>(this->data, tensor.data, this->shape_mult[dimension]);
    return *this;
}


template <typename datatype, size_t dimension, typename comm, typename map>
void DenseTensor<datatype, dimension, comm, map>::insert_value(std::array<size_t, dimension> index, datatype value){
    this->operator()(index) = value;
    return;
}

template <typename datatype, size_t dimension, typename comm, typename map>
std::unique_ptr<DecomposeResult<datatype, dimension, comm, map> > DenseTensor<datatype, dimension, comm, map>::decompose(const std::string method){
    std::cout << method << " is not implemented yet." << std::endl;
    //exit(-1);
}

};
