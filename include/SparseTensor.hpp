#pragma once
#include "Tensor.hpp"
#include "Type.hpp"
#include "device/LinearOp.hpp"
#include <algorithm>

namespace SE{


template<size_t dimension, typename DATATYPE, typename MAPTYPE=Contiguous1DMap<dimension>, DEVICETYPE device=DEVICETYPE::BASE> 
class SparseTensor: public Tensor<dimension, DATATYPE, MAPTYPE, device, STORETYPE::COO > {


using array_d = std::array<size_t, dimension>;
using INTERNALTYPE =  std::vector<std::pair<array_d, DATATYPE> >;
//template <size_t I,typename U,typename T>  using TUPLEDATA = typename tuple_n<I,U,T>::template type<U>;

public:
    SparseTensor(const Comm<device>& _comm, const MAPTYPE& _map);
    SparseTensor(const Comm<device>& _comm, const MAPTYPE& _map, size_t reserve_size);
    SparseTensor(const Comm<device>& _comm, const MAPTYPE& _map, INTERNALTYPE data);

    ~SparseTensor() {
        if (complete_index != nullptr) {
            delete[] complete_index;
            complete_index = nullptr;
        }
        if (complete_value != nullptr) {
            delete[] complete_value;
            complete_value = nullptr;
        }
    }

    INTERNALTYPE copy_data() const override{return this->data;};

    // clone
    SparseTensor<dimension, DATATYPE, MAPTYPE, device>* clone(bool call_complete) const override;
    // insert function
    void global_insert_value(const array_d global_array_index, const DATATYPE value) override;
    void local_insert_value (const array_d local_array_index, const DATATYPE value) override;
    void global_insert_value(const size_t global_index, const DATATYPE value) override;
    void local_insert_value (const size_t local_index, const DATATYPE value) override; 

    void global_set_value(const array_d global_array_index, const DATATYPE value) override;
    void local_set_value (const array_d local_array_index,  const DATATYPE value) override;
    void global_set_value(const size_t global_index,        const DATATYPE value) override;
    void local_set_value (const size_t local_index,         const DATATYPE value) override;
    // string stream
    friend std::ostream& operator<< (std::ostream& stream, const SparseTensor<dimension,DATATYPE,MAPTYPE,device>& tensor){
        stream << static_cast<const Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::COO>&> (tensor);

        stream << "========= Tensor Content =========" <<std::endl;
        auto const num = tensor.get_num_nonzero();
        for(size_t j=0;j<dimension;j++){
            stream << j << '\t';
        }
        stream  << "value" << std::endl;
        stream << "==================================" <<std::endl;
        for (size_t i=0; i<num; i++){
            for(size_t j=0; j<dimension;j++){
                stream << tensor.data[i].first[j] << '\t';
            }
            stream << tensor.data[i].second << std::endl;
        }
        stream << std::endl;
        return stream;
    }

    // Sparse Tensor Only 
    void complete(bool reuse=false);
    
    DATATYPE operator() (const size_t local_index);

    //TUPLEDATA<dimension, DATATYPE*,int*> complete_value; 
    //
    //std::array<dimension, int*> complete_index;
    int* complete_index;
    DATATYPE* complete_value;

    size_t get_num_nonzero() const {return nnz;};

private:
    size_t nnz=0;
};
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,MAPTYPE,device>::SparseTensor(const Comm<device>& comm, const MAPTYPE& map)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::COO>(comm,map){
    this->filled=false;
};

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,MAPTYPE,device>::SparseTensor(const Comm<device>& comm, const MAPTYPE& map, size_t reserve_size)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::COO>(comm,map){
    this->data.reserve(reserve_size);
    this->filled=false;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,MAPTYPE,device>::SparseTensor(const Comm<device>& comm, const MAPTYPE& map, INTERNALTYPE data)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::COO>(comm,map, data){
    this->filled=false;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
SparseTensor<dimension, DATATYPE, MAPTYPE, device>* SparseTensor<dimension,DATATYPE,MAPTYPE,device>::clone(bool call_complete) const{
    auto return_val = new SparseTensor<dimension,DATATYPE,MAPTYPE,device> (*this->copy_comm(), *this->copy_map() );
    if(this->filled && call_complete){
        auto nnz = get_num_nonzero();
        return_val->complete_index = malloc<int,device>(nnz*dimension );
        return_val->complete_value = malloc<DATATYPE,device>(nnz);
        // DEVICE2DEVICE will be ignored if deivce is less than 10
        memcpy(return_val->complete_index, this->complete_index, nnz,COPYTYPE::DEVICE2DEVICE);
        memcpy(return_val->complete_value, this->complete_value, nnz,COPYTYPE::DEVICE2DEVICE);
    }
    else if(this->filled && call_complete==false){
        // if call complete with reuse=false, copy is not available anymore
        assert( this->data.size() == this->get_num_nonzero());
        return_val->data = this->data;
    }
    else{
        return_val->data = this->data;
        if (call_complete) return_val->complete();
    }
    return return_val;
}

// insert function
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::global_insert_value(const std::array<size_t, dimension> global_array_index, const DATATYPE value){
    assert (this->filled==false);
    auto local_array_index = this->map.global_to_local(global_array_index);
    local_insert_value(local_array_index, value);
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::local_insert_value(const std::array<size_t, dimension> local_array_index, const DATATYPE value){
    assert (this->filled==false);
    for(size_t i=0;i<dimension;i++){
        assert (local_array_index[i] >=0 && local_array_index[i]<this->map.get_local_shape(i));
    }
    auto item = std::make_pair(local_array_index, value);
    this->data.push_back(item);
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::global_insert_value(const size_t global_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_insert_value(const size_t global_index, const DATATYPE value) is not implemented");
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::local_insert_value(size_t local_index, DATATYPE value){
    assert (this->filled==false);
    this->data[local_index].second+=value;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::global_set_value(const std::array<size_t, dimension> global_array_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_set_value(const std::array<size_t, dimension> global_index, const DATATYPE value) is not implemented");
}
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::local_set_value (const std::array<size_t, dimension> local_array_index,  const DATATYPE value){
    static_assert(true, "SparseTensor::local_set_value(const std::array<size_t, dimension> global_index, const DATATYPE value) is not implemented");
}
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::global_set_value(const size_t global_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_set_value(const size_t global_index, const DATATYPE value) is not implemented");
}
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::local_set_value (const size_t local_index, const DATATYPE value){
    assert (this->filled==false);
    this->data[local_index].second=value;
}



// Sparse Tensor Only 
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,MAPTYPE,device>::complete(bool reuse){
    if(!this->filled && this->data.size() !=0){

        std::sort(this->data.begin(), this->data.end());
        for (size_t i = 0; i < this->data.size() - 1; i++){
            if(this->data[i].first == this->data[i+1].first){
                this->data[i].second += this->data[i+1].second;
                this->data.erase(std::begin(this->data)+i+1);
                i -= 1;
            }
        }

        auto tmp_index = malloc<int> (this->data.size()*dimension);
        auto tmp_value  = malloc<DATATYPE>(this->data.size());
    
        auto offset= this->data.size();
        for (size_t i = 0; i < this->data.size() ; i++){
            tmp_value[i] = this->data[i].second;
            for (size_t j=0; j<dimension; j++){
                tmp_index[j*offset+i] = this->data[i].first[j];
            }
        }

        nnz = this->data.size();
        if(reuse==false){
            this->data.clear();
        }

        if ( (int)device <10){
            complete_index = malloc<int,device> (nnz*dimension);
            complete_value  = malloc<DATATYPE,device>(nnz);
            memcpy<int,      device>( complete_index, tmp_index, nnz*dimension,  COPYTYPE::HOST2DEVICE);
            memcpy<DATATYPE, device>( complete_value, tmp_value, nnz,            COPYTYPE::HOST2DEVICE);
            free<>(tmp_index);
            free<>(tmp_value);
        }
        else{
            complete_index = tmp_index;
            complete_value = tmp_value;
        }
    }
    this->filled = true;
    return;
};

//// print functions
//template<typename DATATYPE, size_t dimension, DEVICETYPE device, typename MAPTYPE>
//void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::print() const{
//
//    for(auto const &i: this->data){
//        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
//        std::cout << std::setw(6) << i.second << std::endl;
//    }
//    return;
//}
//template<typename DATATYPE, size_t dimension, DEVICETYPE device, typename MAPTYPE>
//void Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::print(const std::string& name) const{
//
//    if(!this->map.is_sliced){
//        if(this->comm->get_rank() == 0){
//            std::cout << name << " : " << std::endl;
//            print();
//            std::cout << "=======================" << std::endl;
//        }
//    }
//    else{
//        std::cout << name << " : (rank " << this->comm->get_rank() << ")" << std::endl;
//        print();
//    }
//}



// get functions
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
DATATYPE SparseTensor<dimension,DATATYPE,MAPTYPE,device>::operator() (const size_t local_index){
//    auto local_array_index = this->map.pack_local_index(local_index);
//    auto iter = std::find(this->data.begin(), this->data.end(), [local_array_index](std::pair<std::array<size_t, dimension>, DATATYPE> element){return element.first==local_array_index;} );
//    if (iter==this->data.end()) return Zero<DATATYPE>::value;
//    return iter->second;
    return this->data[local_index];

};


//template<typename DATATYPE, size_t dimension, DEVICETYPE device, typename MAPTYPE>
//size_t Tensor<STORETYPE::COO, DATATYPE, dimension, device, MAPTYPE>::calculate_column( std::array<size_t, dimension> index, size_t dim){
//    size_t return_val = 0;
//    size_t stride = 1;
//    for (size_t i = 0; i < dimension; i++){
//        const size_t curr_dim = dimension - i - 1;
//        if(curr_dim == dim){
//            continue;
//        }
//        return_val += stride * index[curr_dim];
//        stride *= this->shape[curr_dim];
//    }
//    return return_val;
//};

};
