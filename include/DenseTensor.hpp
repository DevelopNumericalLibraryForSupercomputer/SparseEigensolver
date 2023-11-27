#pragma once
#include <iostream>
#include <array>
#include <cassert>
#include <iomanip>
#include "Tensor.hpp"
#include "Device.hpp"
#include "ContiguousMap.hpp"
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
    Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, datatype* _data);

    datatype& operator()(const std::array<size_t, dimension> index);
    datatype& operator[](size_t index);

    Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>& operator=(const Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype> &tensor);

    void insert_value(std::array<size_t, dimension> index, datatype value);
    void print() const;
    void print(const std::string& name) const;
    Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>* clone() {
        return new Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype> (this->comm, this->map, this->shape, this->data);
    };
    void redistribute(bool slice, size_t slice_dimension);
};

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape): comm(_comm), shape(_shape){
    this->map = _map;
    cumprod<dimension>(this->shape, this->shape_mult);
    assert(this->shape_mult[dimension] != 0);
    size_t datasize = this->shape_mult[dimension];
    if(map->is_sliced){
    
         datasize = map->get_my_partitioned_data_size(comm->rank);//= datasize * map->get_my_partition_size(comm->rank) / shape[map->sliced_dimension];
    }
    
    this->data = malloc<datatype, computEnv>(datasize);
    memset<datatype, computEnv>(this->data, 0.0, datasize);
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::Tensor(Comm<computEnv>* _comm, maptype* _map, std::array<size_t, dimension> _shape, datatype* _data)
:Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>(_comm, _map, _shape){
    if(map->is_sliced){
        memcpy<datatype, computEnv>( this->data, _data, map->get_my_partitioned_data_size(comm->rank));
    }
    else{
        memcpy<datatype, computEnv>( this->data, _data, this->shape_mult[dimension]);
    }
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
    int my_rank = comm->rank;
    size_t target_rank = 0;
    if(map->is_sliced){
        target_rank = map->get_my_rank_from_global_index(index[map->sliced_dimension]);
    }
    if(my_rank == target_rank){
        //std::cout << my_rank << " : (" << index[0] << " , " << index[1] << ") " << map->get_local_index(index, target_rank) << " " << value << std::endl;
        this->data[map->get_local_index(index, target_rank)] = value;
    }
    return;
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::print() const{
    std::cout << "print is not implemented yet." << std::endl;
    exit(-1);
}

template <>
void Tensor<STORETYPE::Dense, double, 2, MKL, ContiguousMap<2> >::print() const{
    for(int i=0;i<this->shape[0];i++){
        for(int j=0;j<this->shape[1];j++){
            std::cout << std::setw(6) << this->data[i+j*this->shape[0]] << " ";
        }
        std::cout << std::endl;
    }
}

template <>
void Tensor<STORETYPE::Dense, double, 1, MKL, ContiguousMap<1> >::print() const{
    for(int i=0;i<this->shape[0];i++){
        std::cout << std::setw(6) << this->data[i] << std::endl;
    }
}

template <>
void Tensor<STORETYPE::Dense, double, 2, MPI, ContiguousMap<2> >::print() const{
    size_t datasize = this->shape_mult[2];
     //std::cout << "datasize " << datasize << " rank = " << comm->rank << std::endl;
    if(map->is_sliced){
        //std::cout << "sliced, " << map->get_my_partition_size(comm->rank) << " / " << shape[map->sliced_dimension] << " rank = " << comm->rank << std::endl;
        //datasize = datasize * map->get_my_partition_size(comm->rank) / shape[map->sliced_dimension];
        datasize = map->get_my_partitioned_data_size(comm->rank);
    }
    //std::cout << "datasize " << datasize << " rank = " << comm->rank << std::endl;
    std::cout << "rank\ti\tj\tvalue" << std::endl;
    for(size_t i=0;i<datasize;i++){
        //std::cout <<  "i : " << i << " rank : " << comm->rank << ", global array : (";
        std::cout <<comm->rank << '\t';
        for(int j=0;j<2;j++){
            //std::cout << comm->rank << ' ' <<  map->get_global_array_index(i,comm->rank)[j] << '\t';
            //std::cout << i << '\t' <<  map->get_global_array_index(i,comm->rank)[j] << '\t';
            std::cout << map->get_global_array_index(i,comm->rank)[j] << '\t';
        }
        //std::cout << ") = " << std::setw(6) << this->data[i] << std::endl;
        std::cout << this->data[i] << std::endl;
    }
}

template <>
void Tensor<STORETYPE::Dense, double, 1, MPI, ContiguousMap<1> >::print() const{
    size_t datasize = this->shape_mult[1];
    if(map->is_sliced){
        datasize = map->get_my_partitioned_data_size(comm->rank);
    }
    std::cout << "rank\ti\tvalue" << std::endl;
    for(size_t i=0;i<datasize;i++){
        std::cout << comm->rank << '\t' << map->get_global_array_index(i,comm->rank)[0] << '\t' << std::setw(6) << this->data[i] << std::endl;
    }
}

template<typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::print(const std::string& name) const{
    if(!map->is_sliced){
        if(this->comm->rank == 0){
            std::cout << name << " : " << std::endl;
            print();
            std::cout << "=======================" << std::endl;
        }
    }
    else{
        std::cout << name << " : (rank " << this->comm->rank << ")" << std::endl;
        print();
    }
}

template <typename datatype, size_t dimension, typename computEnv, typename maptype>
void Tensor<STORETYPE::Dense, datatype, dimension, computEnv, maptype>::redistribute(bool slice, size_t slice_dimension){
    if(this->map->sliced_dimension == slice_dimension && (int)slice == (int) this->map->is_sliced){
        return;
    }
    else{
        //old map info
        this->map->redistribute(slice, slice_dimension);
        //alltoall
    }
}

};
