#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include "Tensor.hpp"
#include "Device.hpp"
//#include "decomposition/Utility.hpp"
//#include "decomposition/DecomposeOption.hpp"
namespace SE{
template<typename datatype, size_t dimension, typename computEnv, typename maptype>
class DenseTensor: public Tensor<datatype, dimension, computEnv, maptype>{
public:
    datatype* data;

    DenseTensor(){};
    DenseTensor(Comm<computEnv>* _comm, Map<dimension>* _map, std::array<size_t, dimension> _shape);
    DenseTensor(Comm<computEnv>* _comm, Map<dimension>* _map, std::array<size_t, dimension> _shape, datatype* data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);

    DenseTensor<datatype, dimension, computEnv, maptype>& operator=(const DenseTensor<datatype, dimension, computEnv, maptype> &tensor);

    //void complete(){};
    //bool get_filled() {return true;};
    void insert_value(std::array<size_t, dimension> index, datatype value);
    void print_tensor();
    DenseTensor<datatype, dimension, computEnv, maptype> clone() {return DenseTensor<datatype, dimension, computEnv, maptype> (this->shape, this->data); };
    /*
    std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > decompose(const std::string method);

    std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > evd();
    std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > davidson();
    void preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess);
    */

};

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension, computEnv, maptype>::DenseTensor(
    Comm<computEnv>* _comm, Map<dimension>* _map, std::array<size_t, dimension> _shape): Tensor<datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->data = malloc<datatype, this->comm::SEenv>(this->shape_mult[dimension]);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension, computEnv, maptype>::DenseTensor(
    Comm<computEnv>* _comm, Map<dimension>* _map, std::array<size_t, dimension> _shape, datatype* data): Tensor<datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    this->data = malloc<datatype, this->comm::SEenv>(this->shape_mult[dimension]);
    memcpy<datatype,this->comm::SEenv>( this->data, data, this->shape_mult[dimension]);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &DenseTensor<datatype, dimension, computEnv, maptype>::operator()(const std::array<size_t, dimension> index){
    // combined index : index[0] + index[1] * dims[0] + index[2] * dimes[0] * dims[1] + ...
    // i.e. 2D matrix : column major, i + row*j
    size_t combined_index = 0;
    for(size_t i = 0; i < dimension; ++i){
        combined_index += index[i] * this->shape_mult[i];
    }
    return this->data[combined_index];
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
datatype &DenseTensor<datatype, dimension, computEnv, maptype>::operator[](size_t index){
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

template <class datatype, size_t dimension, typename computEnv, typename maptype>
DenseTensor<datatype, dimension, computEnv, maptype>& DenseTensor<datatype, dimension, computEnv, maptype>::operator=(const DenseTensor<datatype, dimension, computEnv, maptype>& tensor){
    if(this == &tensor){
        return *this;
    }
    this->comm = tensor.comm;
    this->map = tensor.map;
    this->shape = tensor.shape;
    this->shape_mult = tensor.shape_mult;
    // cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0 );

    //delete[] this->data;
    //this->data = new datatype[this->shape_mult[dimension]];
    free<datatype, computEnv>(this->data);
    this->data = malloc<datatype, computEnv>(this->shape_mult[dimension]);
    memcpy<datatype, computEnv>(this->data, tensor.data, this->shape_mult[dimension]);
    return *this;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void DenseTensor<datatype, dimension, computEnv, maptype>::insert_value(std::array<size_t, dimension> index, datatype value){
    this->operator()(index) = value;
    return;
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void DenseTensor<datatype, dimension, computEnv, maptype>::print_tensor(){
        std::cout << "print is not implemented yet." << std::endl;
        exit(-1);
    
}

template <typename datatype, typename computEnv, typename maptype>
void DenseTensor<datatype, 2, computEnv, maptype>::print_tensor(){
    
        std::cout << "=======================" << std::endl;
        for(int i=0;i<this->shape[0];i++){
            for(int j=0;j<this->shape[1];j++){
                std::cout << std::setw(6) << this->data[i+j*this->shape[1]] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "=======================" << std::endl;
    
}
/*

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > DenseTensor<datatype, dimension, computEnv, maptype>::decompose(const std::string method){
    if(method.compare("EVD")==0){
        return evd();
    }
    else if(method.compare("Davidson")==0){
        return davidson();
    }
    else{
        std::cout << method << " is not implemented yet." << std::endl;
        exit(-1);
    }
    
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > DenseTensor<datatype, dimension, computEnv, maptype>::evd(){
    std::cout << "EVD is not implemented yet." << std::endl;
    exit(-1);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
std::unique_ptr<DecomposeResult<datatype, dimension, computEnv, maptype> > DenseTensor<datatype, dimension, computEnv, maptype>::davidson(){
    std::cout << "davidson is not implemented yet." << std::endl;
    exit(-1);
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void DenseTensor<datatype, dimension, computEnv, maptype>::preconditioner(DecomposeOption option, double* sub_eigval, double* residual, size_t block_size, double* guess){
    std::cout << "invalid preconditioner." << std::endl;
    exit(-1);
}
*/
};
