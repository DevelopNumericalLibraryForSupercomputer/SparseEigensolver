#pragma once
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"

#include "../device/TensorOp.hpp"


namespace SE{
template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device, STORETYPE store> 
class TensorOperations{
public:
    TensorOperations(){};
    template<size_t vec_dimension> 
    DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device> matvec(const DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device>& vec){return vec;};

    virtual DATATYPE get_diag_element(const size_t index) = 0;

    virtual std::array<size_t, dimension> get_global_shape() =0;
    virtual Comm<device> get_comm() =0;
    virtual Comm<device>* copy_comm() =0;
    virtual MAPTYPE get_map() =0;
};

template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
class BasicDenseTensorOperations: public TensorOperations<dimension, DATATYPE, MAPTYPE, device, STORETYPE::DENSE>{
    //default class
public:
    BasicDenseTensorOperations(DenseTensor<dimension, DATATYPE, MAPTYPE, device>& tensor):tensor(tensor){};

    template<size_t vec_dimension> 
    DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device> matvec(const DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device>& vec){
        return TensorOp::matmul(this->tensor, vec);
    };

    DATATYPE get_diag_element(const size_t index) override{
        std::array<size_t, 2> array_index = {index, index};
        return tensor.operator()(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
    };

    std::array<size_t, dimension> get_global_shape() override{
        return tensor.map.get_global_shape();
    };
    Comm<device> get_comm() override{
        return tensor.comm;
    }
    Comm<device>* copy_comm() override{
        return tensor.copy_comm();
    }
    MAPTYPE get_map() override{
        return tensor.map;
    }
private:
    DenseTensor<dimension, DATATYPE, MAPTYPE, device> tensor;
};




template<size_t dimension, typename DATATYPE, typename MAPTYPE, DEVICETYPE device> 
class BasicSparseTensorOperations: public TensorOperations<dimension, DATATYPE, MAPTYPE, device, STORETYPE::COO>{
    //default class
public:
    BasicSparseTensorOperations(SparseTensor<dimension, DATATYPE, MAPTYPE, device>& tensor):tensor(tensor){};

    template<size_t vec_dimension> 
    DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device> matvec(const DenseTensor<vec_dimension, DATATYPE, MAPTYPE, device>& vec){
        return matmul(this->tensor, vec);
    };

    DATATYPE get_diag_element(const size_t index) override{
        std::array<size_t, 2> array_index = {index, index};
        return tensor.operator()(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
    };

    std::array<size_t, dimension> get_global_shape() override{
        return tensor.map.get_global_shape();
    };
    Comm<device> get_comm() override{
        return tensor.comm;
    }
    Comm<device>* copy_comm() override{
        return tensor.copy_comm();
    }
    MAPTYPE get_map() override{
        return tensor.map;
    }
private:
    SparseTensor<dimension, DATATYPE, MAPTYPE, device> tensor;
};

}