#pragma once
#include "Tensor.hpp"
#include "Contiguous1DMap.hpp"
#include "device/LinearOp.hpp"

namespace SE{

//template<int dimension, typename DATATYPE, typename MAPTYPE=Contiguous1DMap<dimension>, DEVICETYPE DEVICETYPE=DEVICETYPE::BASE> 
template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
class DenseTensor: public Tensor<dimension, DATATYPE, MAPTYPE, DEVICETYPE, STORETYPE::DENSE > {

using array_d = std::array<int, dimension>;
//using INTERNALTYPE = std::conditional< store==STORETYPE::DENSE,  DATATYPE* , std::vector<std::pair<array_d, DATATYPE> > >; 
using INTERNALTYPE = DATATYPE*;

public:
    DenseTensor();
    DenseTensor(const Comm<DEVICETYPE>& comm, const MAPTYPE& map);
    DenseTensor(const Comm<DEVICETYPE>& comm, const MAPTYPE& map, INTERNALTYPE data);
    DenseTensor(const DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>& tensor);

    DATATYPE* copy_data() const override;

    DenseTensor<dimension, DATATYPE, MAPTYPE, DEVICETYPE>* clone(bool call_complete) const override{
        auto return_val = new DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>(*this->copy_comm(), *this->copy_map(), this->copy_data() );
        if(call_complete) return_val->complete();
        return return_val;
    }

    void global_insert_value(const array_d global_array_index, const DATATYPE value) override;
    void local_insert_value(const array_d local_array_index, const DATATYPE value) override;
    void global_insert_value(const int global_index, const DATATYPE value) override;
    void local_insert_value(const int local_index, const DATATYPE value) override;

    void global_set_value(const array_d global_array_index, const DATATYPE value) override;
    void local_set_value(const array_d local_array_index, const DATATYPE value) override;
    void global_set_value(const int global_index, const DATATYPE value) override;
    void local_set_value(const int local_index, const DATATYPE value) override;

    friend std::ostream& operator<< (std::ostream& stream, const DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>& tensor){
        stream << static_cast<const Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>&> (tensor);
        tensor.comm.barrier();
        for (int rank=0; rank<tensor.comm.get_world_size(); rank++){
            if(rank!=tensor.comm.get_rank()){
                tensor.comm.barrier();
            }
            else{
                stream << "========= Tensor Content"<< rank << "=========" <<std::endl;
                if(dimension == 1){
                    auto const num_row = tensor.map.get_local_shape(0);
                    for (int i=0; i<num_row; i++){
                        stream << tensor.data[i] << " ";
                    }
                    stream << std::endl;
                }
                else if (dimension == 2){
                    auto const num_row = tensor.map.get_local_shape(0);
                    auto const num_col = tensor.map.get_local_shape(1);
                    for (int i=0; i<num_row; i++){
                        for (int j=0; j<num_col; j++){
                            stream << std::fixed << std::setw(5) << std::setprecision(2) << tensor.data[i*num_col + j] << " ";
                        }
                        stream << std::endl;
                    }
                }
                else{
                    for(int j=0;j<dimension;j++){
                        stream << j << '\t';
                    }
                    stream  << "value" << std::endl;
                    stream  << "=================================" << std::endl;
                    auto const num = tensor.map.get_num_local_elements();
                    for (int i=0; i<num; i++){
                        auto global_index_array_tmp = tensor.map.local_to_global(tensor.map.pack_local_index(i));
                        for(int j=0; j<dimension;j++){
                            stream << global_index_array_tmp[j] << '\t';
                        }
                        stream << tensor.data[i] << " ";
                    }
                }
                tensor.comm.barrier();
            }
        }
        return stream;
    }
};

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::DenseTensor(const Comm<DEVICETYPE>& comm, const MAPTYPE& map)
:Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>(comm,map){
    auto data_size = this->map.get_num_local_elements();
    this->data = malloc<DATATYPE, DEVICETYPE>( data_size );
    memset<DATATYPE,DEVICETYPE>( this->data, 0, data_size);
    this->filled=false;
};


template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::DenseTensor(const Comm<DEVICETYPE>& comm, const MAPTYPE& map, INTERNALTYPE data)
:Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>(comm,map,data){
    this->filled=false;
};

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::DenseTensor(const DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>& tensor)
:Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>(tensor){};


template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
DATATYPE* DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::copy_data() const{
    DATATYPE* return_data;
    auto data_size = this->map.get_num_local_elements();
    return_data = malloc<DATATYPE, DEVICETYPE>( data_size );
    memcpy<DATATYPE, DEVICETYPE>(return_data, this->data, data_size);
    return return_data;
}


template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::global_insert_value(const std::array<int, dimension> global_array_index, const DATATYPE value){
    int local_index = this->map.unpack_local_array_index(this->map.global_to_local(global_array_index));
    local_insert_value(local_index, value);
    return;
}

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::local_insert_value(const std::array<int, dimension> local_array_index, const DATATYPE value){
    for(int i=0;i<dimension;i++){
        assert (local_array_index[i] < this->map.get_local_shape(i));
    }
    local_insert_value(this->map.unpack_local_array_index(local_array_index), value);
    return;
}

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::global_insert_value(const int global_index, const DATATYPE value){
    int local_index = this->map.global_to_local(global_index);
    local_insert_value(local_index, value);
    return;
}

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::local_insert_value(const int local_index, const DATATYPE value) {
    assert (local_index <this->map.get_num_local_elements());
	if(local_index<0) return;
    this->data[local_index] += value;
    return;
}

template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::global_set_value(std::array<int, dimension> global_array_index, DATATYPE value){
    int local_index = this->map.unpack_local_array_index(this->map.global_to_local(global_array_index));
    local_set_value(local_index, value);
    return;
}
template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::local_set_value(std::array<int, dimension> local_array_index, DATATYPE value) {
    for(int i=0;i<dimension;i++){
        assert ( local_array_index[i] < this->map.get_local_shape(i));
    }
    local_set_value(this->map.unpack_local_array_index(local_array_index), value);
    return;
}
template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::global_set_value(int global_index, DATATYPE value) {
    int local_index = this->map.global_to_local(global_index);
    local_set_value(local_index, value);
}
template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE> 
void DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>::local_set_value(int local_index, DATATYPE value) {
    assert (local_index <this->map.get_num_local_elements());
	if(local_index<0) return;
    this->data[local_index] = value;
    return;
}

/*
template<int dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE>
std::ostream& operator<< (std::ostream& stream, const DenseTensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE>& tensor){
    //std::cout <<(Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>)tensor << std::endl;
    //const Tensor<2,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>*  p_tensor = &tensor;
    //std::cout <<*p_tensor <<std::endl;

    tensor.print_tensor_info();

    auto const num_row = tensor.map.get_global_shape(0);
    auto const num_col = tensor.map.get_global_shape(1);

    stream << "========= Tensor Content =========" <<std::endl;
    for (int i=0; i<num_row; i++){
        for (int j=0; j<num_col; j++){
            stream << tensor.data[i+j*num_row] << " ";
        }
        stream << std::endl;
    }
    return stream;
}

template<typename DATATYPE, typename MAPTYPE, DEVICETYPE DEVICETYPE>
std::ostream& operator<< (std::ostream& stream, const DenseTensor<1,DATATYPE,MAPTYPE,DEVICETYPE>& tensor){
    //std::cout <<(Tensor<dimension,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>)tensor << std::endl;
    //const Tensor<1,DATATYPE,MAPTYPE,DEVICETYPE,STORETYPE::DENSE>*  p_tensor = &tensor;
    //std::cout <<*p_tensor <<std::endl;
    stream << static_cast<const Tensor<1, DATATYPE, MAPTYPE, DEVICETYPE, STORETYPE::DENSE> &> (tensor);

    auto const num_row = tensor.map.get_global_shape(0);

    stream << "========= Tensor Content =========" <<std::endl;
    for (int i=0; i<num_row; i++){
        stream << tensor.data[i] << " ";
    }
    stream << std::endl;
    return stream;
}
*/
}
