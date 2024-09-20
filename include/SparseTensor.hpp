#pragma once
#include "Tensor.hpp"
#include "Type.hpp"
#include "device/LinearOp.hpp"
#include "Contiguous1DMap.hpp"
#include <algorithm>

namespace SE{


template<int dimension, typename DATATYPE, MTYPE mtype=MTYPE::Contiguous1D, DEVICETYPE device=DEVICETYPE::BASE> 
class SparseTensor: public Tensor<dimension, DATATYPE, mtype, device, STORETYPE::COO > {


using array_d = std::array<int, dimension>;
using INTERNALTYPE =  std::vector<std::pair<array_d, DATATYPE> >;
//template <int I,typename U,typename T>  using TUPLEDATA = typename tuple_n<I,U,T>::template type<U>;

public:
    SparseTensor();
    SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map);
    SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map, int reserve_size);
    SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map, INTERNALTYPE data);

    ~SparseTensor() {}
/*
    //copy assign operator
    SparseTensor<dimension,DATATYPE,mtype,device>& SparseTensor<dimension,DATATYPE,mtype,device>::operator=(const SparseTensor<dimension,DATATYPE,mtype,device>& other){
        if(this == &other) return *this;
    
        this->ptr_comm = other->copy_comm();
        this->ptr_map = other->copy_map();
        
        if(other->filled){
            auto nnz = other->get_num_nonzero();
            this->complete_index = malloc<int,device>(nnz*dimension );
            this->complete_value = malloc<DATATYPE,device>(nnz);
            // DEVICE2DEVICE will be ignored if deivce is less than 10
            memcpy(this->complete_index, other->complete_index, nnz,COPYTYPE::DEVICE2DEVICE);
            memcpy(this->complete_value, other->complete_value, nnz,COPYTYPE::DEVICE2DEVICE);
        }
        else if(other->filled && call_complete==false){
            // if call complete with reuse=false, copy is not available anymore
            assert( other->data.size() == other->get_num_nonzero());
            this->data = other->data;
        }
        else{
            this->data = other->data;
        }
        this->filled = other->filled;
    
        return *this;
    };

    //move assign operator
    SparseTensor<dimension,DATATYPE,mtype,device>& SparseTensor<dimension,DATATYPE,mtype,device>::operator=(SparseTensor<dimension,DATATYPE,mtype,device>&& other){
        if(this == &other) return *this;
        this->ptr_comm = std::move(other.ptr_comm);
        this->ptr_map = std::move(other.ptr_map);
        this->data = std::move(other.data);
        this->filled = other.filled;
        if(other->filled){
            //move pointers
            this->complete_index = other->complete_index;
            this->complete_value = other->complete_value;
        }
        return *this;
    };
*/
    INTERNALTYPE copy_data() const override{return this->data;};

    // clone
    std::unique_ptr<SparseTensor<dimension, DATATYPE, mtype, device> > clone(bool call_complete) const;
    // insert function
    void global_insert_value(const array_d global_array_index, const DATATYPE value) override;
    void local_insert_value (const array_d local_array_index, const DATATYPE value) override;
    void global_insert_value(const int global_index, const DATATYPE value) override;
    void local_insert_value (const int local_index, const DATATYPE value) override; 

    void global_set_value(const array_d global_array_index, const DATATYPE value) override;
    void local_set_value (const array_d local_array_index,  const DATATYPE value) override;
    void global_set_value(const int global_index,        const DATATYPE value) override;
    void local_set_value (const int local_index,         const DATATYPE value) override;
    // string stream
    friend std::ostream& operator<< (std::ostream& stream, const SparseTensor<dimension,DATATYPE,mtype,device>& tensor){
        stream << static_cast<const Tensor<dimension,DATATYPE,mtype,device,STORETYPE::COO>&> (tensor);

        stream << "========= Tensor Content =========" <<std::endl;
        auto const num = tensor.get_num_nonzero();
        for(int j=0;j<dimension;j++){
            stream << j << '\t';
        }
        stream  << "value" << std::endl;
        stream << "==================================" <<std::endl;
        for (int i=0; i<num; i++){
            for(int j=0; j<dimension;j++){
                stream << tensor.data[i].first[j] << '\t';
            }
            stream << tensor.data[i].second << std::endl;
        }
        stream << std::endl;
        return stream;
    }

    // Sparse Tensor Only 
    void complete(bool reuse=true);
    
    DATATYPE operator() (const int local_index) const { return this->data[local_index]; };

    //TUPLEDATA<dimension, DATATYPE*,int*> complete_value; 
    //
    //std::array<dimension, int*> complete_index;
    std::unique_ptr<int[], std::function<void(int*)> > complete_index;
    std::unique_ptr<DATATYPE[], std::function<void(DATATYPE*)> > complete_value;

    int get_num_nonzero() const {return nnz;};

private:
    int nnz=0;
};
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,mtype,device>::SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map)
:Tensor<dimension,DATATYPE,mtype,device,STORETYPE::COO>(ptr_comm,ptr_map){
    this->filled=false;
};

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,mtype,device>::SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map, int reserve_size)
:Tensor<dimension,DATATYPE,mtype,device,STORETYPE::COO>(ptr_comm,ptr_map){
    this->data.reserve(reserve_size);
    this->filled=false;
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
SparseTensor<dimension,DATATYPE,mtype,device>::SparseTensor(const std::unique_ptr<Comm<device>>& ptr_comm, const std::unique_ptr<Map<dimension,mtype>>& ptr_map, INTERNALTYPE data)
:Tensor<dimension,DATATYPE,mtype,device,STORETYPE::COO>(ptr_comm,ptr_map, data){
    this->filled=false;
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device>
std::unique_ptr<SparseTensor<dimension, DATATYPE, mtype, device> > SparseTensor<dimension,DATATYPE,mtype,device>::clone(bool call_complete) const{
    auto return_val = std::make_unique< SparseTensor<dimension,DATATYPE,mtype,device> > (this->copy_comm(), this->copy_map() ); // I am not sure if pointer of inherit class work well shchoi
    if(this->filled && call_complete){
        auto nnz = get_num_nonzero();
        //return_val->complete_index = std::make_unique<int[], std::function<void(int*)>>( malloc<int,device>(nnz*dimension ), free<device> );
        //return_val->complete_value = std::make_unique<DATATYPE[], std::function<void(DATATYPE*)>>( malloc<DATATYPE,device>(nnz), free<device> );

        // DEVICE2DEVICE will be ignored if deivce is less than 10
        memcpy(return_val->complete_index.get(), this->complete_index.get(), nnz,COPYTYPE::DEVICE2DEVICE);
        memcpy(return_val->complete_value.get(), this->complete_value.get(), nnz,COPYTYPE::DEVICE2DEVICE);
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
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::global_insert_value(const std::array<int, dimension> global_array_index, const DATATYPE value){
    assert (this->filled==false);
    auto local_array_index = this->ptr_map->global_to_local(global_array_index);
    local_insert_value(local_array_index, value);
    return;
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::local_insert_value(const std::array<int, dimension> local_array_index, const DATATYPE value){
    assert (this->filled==false);
    for(int i=0;i<dimension;i++){
        assert (local_array_index[i] >=0 && local_array_index[i]<this->ptr_map->get_local_shape(i));
    }
    auto item = std::make_pair(local_array_index, value);
    this->data.push_back(item);
    return;
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::global_insert_value(const int global_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_insert_value(const int global_index, const DATATYPE value) is not implemented");
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::local_insert_value(int local_index, DATATYPE value){
    assert (this->filled==false);
    this->data[local_index].second+=value;
}

template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::global_set_value(const std::array<int, dimension> global_array_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_set_value(const std::array<int, dimension> global_index, const DATATYPE value) is not implemented");
}
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::local_set_value (const std::array<int, dimension> local_array_index,  const DATATYPE value){
    static_assert(true, "SparseTensor::local_set_value(const std::array<int, dimension> global_index, const DATATYPE value) is not implemented");
}
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::global_set_value(const int global_index, const DATATYPE value){
    static_assert(true, "SparseTensor::global_set_value(const int global_index, const DATATYPE value) is not implemented");
}
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::local_set_value (const int local_index, const DATATYPE value){
    assert (this->filled==false);
    this->data[local_index].second=value;
}



// Sparse Tensor Only 
template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
void SparseTensor<dimension,DATATYPE,mtype,device>::complete(bool reuse){
    if(!this->filled && this->data.size() !=0){

        // remove duplicated index and add values
        std::sort(this->data.begin(), this->data.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        // std::sort(this->data.begin(), this->data.end() );
        for (int i = 0; i < this->data.size() - 1; i++){
            if(this->data[i].first == this->data[i+1].first){
                this->data[i].second += this->data[i+1].second;
                this->data.erase(std::begin(this->data)+i+1);
                i -= 1;
            }
        }

        auto tmp_index = malloc<int> (this->data.size()*dimension);
        auto tmp_value  = malloc<DATATYPE>(this->data.size());
    
        const auto offset= this->data.size();
        #pragma omp parallel for
        for (int i = 0; i < this->data.size() ; i++){
            tmp_value[i] = this->data[i].second;
            for (int j=0; j<dimension; j++){
                tmp_index[j*offset+i] = this->data[i].first[j];
            }
        }

        nnz = this->data.size();
        if(reuse==false){
            this->data.clear();
        }

        if ( (int)device <10){
            std::unique_ptr<int[], std::function<void(int*)>> complete_index_( malloc<int,device>(nnz*dimension ), free<device> );
            complete_index = std::move(complete_index_);
            std::unique_ptr<DATATYPE[], std::function<void(DATATYPE*)>> complete_value_( malloc<DATATYPE,device>(nnz), free<device> );
            complete_value = std::move(complete_value_);
            memcpy<int,      device>( complete_index.get(), tmp_index, nnz*dimension,  COPYTYPE::HOST2DEVICE);
            memcpy<DATATYPE, device>( complete_value.get(), tmp_value, nnz,            COPYTYPE::HOST2DEVICE);
        }
        else{
            std::cout << "not yet implemented" <<std::endl;
            exit(-1);
        }
        free<device>(tmp_index);
        free<device>(tmp_value);
    }
    this->filled = true;
    return;
};

//// print functions
//template<typename DATATYPE, int dimension, DEVICETYPE device, MTYPE mtype>
//void Tensor<STORETYPE::COO, DATATYPE, dimension, device, Map<dimension,mtype>>::print() const{
//
//    for(auto const &i: this->data){
//        for(int j=0;j<2;j++) std::cout << i.first[j] << '\t';
//        std::cout << std::setw(6) << i.second << std::endl;
//    }
//    return;
//}
//template<typename DATATYPE, int dimension, DEVICETYPE device, MTYPE mtype>
//void Tensor<STORETYPE::COO, DATATYPE, dimension, device, Map<dimension,mtype>>::print(const std::string& name) const{
//
//    if(!this->ptr_map.is_sliced){
//        if(this->ptr_comm->get_rank() == 0){
//            std::cout << name << " : " << std::endl;
//            print();
//            std::cout << "=======================" << std::endl;
//        }
//    }
//    else{
//        std::cout << name << " : (rank " << this->ptr_comm->get_rank() << ")" << std::endl;
//        print();
//    }
//}



// get functions
//template<int dimension, typename DATATYPE, MTYPE mtype, DEVICETYPE device> 
//DATATYPE SparseTensor<dimension,DATATYPE,mtype,device>::operator() const (const int local_index){
//    return this->data[local_index];
//};


//template<typename DATATYPE, int dimension, DEVICETYPE device, MTYPE mtype>
//int Tensor<STORETYPE::COO, DATATYPE, dimension, device, Map<dimension,mtype>>::calculate_column( std::array<int, dimension> index, int dim){
//    int return_val = 0;
//    int stride = 1;
//    for (int i = 0; i < dimension; i++){
//        const int curr_dim = dimension - i - 1;
//        if(curr_dim == dim){
//            continue;
//        }
//        return_val += stride * index[curr_dim];
//        stride *= this->shape[curr_dim];
//    }
//    return return_val;
//};

};
