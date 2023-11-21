#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Tensor.hpp"
#include "Device.hpp"
//#include "device/MKL/LinearOp.hpp"
namespace SE{
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
class Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>{
public:
    const STORETYPE store_type = STORETYPE::Dense;
    std::array<size_t, dimension> shape;
    std::array<size_t, dimension+1> shape_mult;

    datatype* data;
    Comm<computEnv>* comm;
    maptype* map;

    Tensor(){};
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape);
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, datatype* data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);

    Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>& operator=(const Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype> &tensor);

    void insert_value(std::array<size_t, dimension> index, datatype value);
    void print_tensor();
    Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype> clone() {
        return Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype> (this->comm, this->map, this->shape, this->data);
    };
};

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape): comm(_comm), shape(_shape){
    this->map = _map;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    this->data = malloc<datatype, computEnv>(this->shape_mult[dimension]);
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, datatype* data)
:Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    memcpy<datatype, computEnv>( this->data, data, this->shape_mult[dimension]);
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::operator()(const std::array<size_t, dimension> index){
    // combined index : index[0] + index[1] * dims[0] + index[2] * dimes[0] * dims[1] + ...
    // i.e. 2D matrix : column major, i + row*j
    size_t combined_index = 0;
    for(size_t i = 0; i < dimension; ++i){
        combined_index += index[i] * this->shape_mult[i];
    }
    return this->data[combined_index];
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::operator[](size_t index){
    return this->data[index];
}
/*
template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension, computEnv, maptype> operator+(DenseTensor<datatype, dimension, computEnv, maptype>& a, DenseTensor<datatype, dimension, computEnv, maptype>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), 1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension> operator-(DenseTensor<datatype, dimension, computEnv, maptype>& a, DenseTensor<datatype, dimension, computEnv, maptype>& b){
    if (a.shape != b.shape){
        std::cout << "Can't subtract tensor having different shape." << std::endl;
        exit(-1);
    }
    auto result = a.clone();
    axpy<datatype>(b.data.size(), -1.0, b.data.data(), 1, result.data.data(), 1 );
    
    return result;
}
*/

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>& Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::operator=
        (const Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>& tensor){
    if(this == &tensor){
        return *this;
    }
    this->comm = tensor.comm;
    this->map = tensor.map;
    this->shape = tensor.shape;
    this->shape_mult = tensor.shape_mult;
    assert(this->shape_mult[dimension] != 0 );

    free<datatype, computEnv>(this->data);
    this->data = malloc<datatype, computEnv>(this->shape_mult[dimension]);
    memcpy<datatype, computEnv>(this->data, tensor.data, this->shape_mult[dimension]);
    return *this;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::insert_value(std::array<size_t, dimension> index, datatype value){
    this->operator()(index) = value;
    return;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::print_tensor(){
    std::cout << "print is not implemented yet." << std::endl;
    exit(-1);
}

template <>
void Tensor<STORETYPE::Dense, double, 2, MKL, ContiguousMap<2> >::print_tensor(){

    std::cout << "=======================" << std::endl;
    for(int i=0;i<this->shape[0];i++){
        for(int j=0;j<this->shape[1];j++){
            std::cout << std::setw(6) << this->data[i+j*this->shape[1]] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "=======================" << std::endl;
    
}

template <>
void Tensor<STORETYPE::Dense, double, 2, MPI, ContiguousMap<2> >::print_tensor(){
    if(this->comm->rank == 0){
        std::cout << "=======================" << std::endl;
        for(int i=0;i<this->shape[0];i++){
            for(int j=0;j<this->shape[1];j++){
                std::cout << std::setw(6) << this->data[i+j*this->shape[1]] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    }
}

};
