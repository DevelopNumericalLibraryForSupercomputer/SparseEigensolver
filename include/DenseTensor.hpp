#pragma once
#include "Tensor.hpp"
#include "device/LinearOp.hpp"

namespace SE{

//template<size_t dimension, typename DATATYPE, typename MAPTYPE=Contiguous1DMap<dimension>, DEVICETYPE device=DEVICETYPE::BASE> 
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
class DenseTensor: public Tensor<dimension, DATATYPE, MAPTYPE, device, STORETYPE::DENSE > {

using array_d = std::array<size_t, dimension>;
//using _internal_DATATYPE = std::conditional< store==STORETYPE::DENSE,  DATATYPE* , std::vector<std::pair<array_d, DATATYPE> > >; 
using _internal_DATATYPE = DATATYPE*;

public:
    DenseTensor(const Comm<device>& comm, const MAPTYPE& map);
    DenseTensor(const Comm<device>& comm, const MAPTYPE& map, _internal_DATATYPE data);
    DenseTensor(const DenseTensor<dimension,DATATYPE,MAPTYPE,device>& tensor);

    DATATYPE* copy_data() const;

    DenseTensor<dimension, DATATYPE, MAPTYPE, device>* clone(bool call_complete) const {
        auto return_val = new DenseTensor<dimension,DATATYPE,MAPTYPE,device>(*this->copy_comm(), *this->copy_map(), this->copy_data() );
        if(call_complete) return_val->complete();
        return return_val;
    }

    void global_insert_value(array_d global_array_index, DATATYPE value);
    void local_insert_value(array_d local_array_index, DATATYPE value);
    void global_insert_value(size_t global_index, DATATYPE value);
    void local_insert_value(size_t local_index, DATATYPE value);

    friend std::ostream& operator<< <>(std::ostream& stream, const DenseTensor<dimension,DATATYPE,MAPTYPE,device>& tensor);
};

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
DenseTensor<dimension,DATATYPE,MAPTYPE,device>::DenseTensor(const Comm<device>& comm, const MAPTYPE& map)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::DENSE>(comm,map){
    auto data_size = this->map.get_num_local_elements();
    this->data = malloc<DATATYPE, device>( data_size );
    memset<DATATYPE,device>( this->data, data_size, 0);
    this->filled=false;
};


template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
DenseTensor<dimension,DATATYPE,MAPTYPE,device>::DenseTensor(const Comm<device>& comm, const MAPTYPE& map, _internal_DATATYPE data)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::DENSE>(comm,map,data){
    this->filled=false;
};

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
DenseTensor<dimension,DATATYPE,MAPTYPE,device>::DenseTensor(const DenseTensor<dimension,DATATYPE,MAPTYPE,device>& tensor)
:Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::DENSE>(tensor){};


template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
DATATYPE* DenseTensor<dimension,DATATYPE,MAPTYPE,device>::copy_data() const{
    DATATYPE* return_data;
    auto data_size = this->map.get_num_local_elements();
    return_data = malloc<DATATYPE, device>( data_size );
    memcpy<DATATYPE, device>(return_data, this->data, data_size);
    return return_data;
}


template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,device>::global_insert_value(std::array<size_t, dimension> global_array_index, DATATYPE value){
    size_t local_index = this->map.unpack_local_array_index(this->map.global_to_local(global_array_index));
    local_insert_value(local_index, value);
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,device>::local_insert_value(std::array<size_t, dimension> local_array_index, DATATYPE value){
    for(size_t i=0;i<dimension;i++){
        assert (local_array_index[i] >=0 && local_array_index[i] < this->map.get_local_shape(i));
    }
    local_insert_value(this->map.unpack_local_array_index(local_array_index), value);
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,device>::global_insert_value(size_t global_index, DATATYPE value){
    size_t local_index = this->map.global_to_local(global_index);
    local_insert_value(local_index, value);
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,device>::local_insert_value(size_t local_index, DATATYPE value) {
    assert (local_index >=0);
    assert (local_index <this->map.get_num_local_elements());
    this->data[local_index] += value;
    return;
}

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device>
std::ostream& operator<< (std::ostream& stream, const DenseTensor<dimension,DATATYPE,MAPTYPE,device>& tensor){
    //std::cout <<(Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::DENSE>)tensor << std::endl;
    const Tensor<dimension,DATATYPE,MAPTYPE,device,STORETYPE::DENSE>*  p_tensor = &tensor;
    std::cout <<*p_tensor <<std::endl;

    std::cout << "========= Tensor Content =========" <<std::endl;
    auto const num_row = tensor.map.get_global_shape(0);
    auto const num_col = tensor.map.get_global_shape(1);

    for (size_t i=0; i<num_row; i++){
        for (size_t j=0; j<num_col; j++){
            std::cout << tensor.data[i+j*num_row] << " ";
        }
        std::cout << std::endl;
    }
    return stream;
}


}
