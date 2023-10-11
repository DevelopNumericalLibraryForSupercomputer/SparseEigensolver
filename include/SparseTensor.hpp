#pragma once
#include <iostream>
#include <array>
#include <vector>
#include <algorithm>
#include "Tensor.hpp"

namespace SE{

//COO sparse matrix
template<typename datatype, size_t dimension, typename comm, typename map>
class SparseTensor: public Tensor<datatype, dimension, comm, map>{
public:
    //vector of (index array, data)
    std::vector<std::pair<std::array<size_t, dimension>, datatype> > data;

    SparseTensor(){filled = false;};
    SparseTensor(std::array<size_t, dimension> shape);
    SparseTensor(std::array<size_t, dimension> shape, std::vector<std::pair<std::array<size_t, dimension>, datatype> > data);

    datatype& operator()(const std::array<size_t, dimension> index);
    SparseTensor<datatype, dimension, comm, map>& operator=(const SparseTensor<datatype, dimension, comm, map> &tensor);

    void complete();
    bool get_filled(){return filled};
    void insert_value(std::array<size_t, dimension> index, datatype value);

    SparseTensor<datatype, dimension, comm, map> clone() {return SparseTensor<datatype, dimension, comm, map> (this->shape, this->data); };
    DecomposeResult<datatype, dimension, comm> decompose(const char* method);

    void CSR_to_COO();
    void COO_to_CSR();
private:
    bool filled;
};

template <typename datatype, size_t dimension, typename comm, typename map>
SparseTensor<datatype, dimension, comm, map>::SparseTensor(std::array<size_t, dimension> shape){
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    this->filled = false;
}

template <typename datatype, size_t dimension, typename comm, typename map>
SparseTensor<datatype, dimension, comm, map>::SparseTensor(std::array<size_t, dimension> shape, std::vector<std::pair<std::array<size_t, dimension>, datatype> > data)
    : SparseTensor(shape){
    /*
    this->shape = shape;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    */
    this->data = data;
    this->filled = true;
}

template <typename datatype, size_t dimension, typename comm, typename map>
datatype &SparseTensor<datatype, dimension, comm, map>::operator()(const std::array<size_t, dimension> index){
    for (size_t i = 0; i < this->data.size(); i++){
        // array equal, c++20
        //std::cout << i << " " << data[i].first[0] << " " << data[i].first[1] << " " << data[i].first[2] << " " << data[i].second << std::endl;
        if(data[i].first == index){
            return this->data[i].second;
        }
    }
    std::cout << "data not found in this tensor." << std::endl;
    exit(-1);
}

template <typename datatype, size_t dimension, typename comm, typename map>
inline void SparseTensor<datatype, dimension, comm, map>::insert_value(std::array<size_t, dimension> index, datatype value){
    assert(this->filled == false);
    this->data.push_back(std::make_pair(index, value));
}

template<typename datatype, size_t dimension, typename comm, typename map>
void SparseTensor<datatype, dimension, comm, map>::complete()
{
    if(!this->filled && this->data.size() !=0){
        std::sort(this->data.begin(), this->data.end());
        
        for (size_t i = 0; i < data.size() - 1; i++){
            if(data[i].first == data[i+1].first){
                data[i].second += data[i+1].second;
                data.erase(std::begin(data)+i+1);
                
                i -= 1;
            }
        }
    }
    this->filled = true;
}

}
