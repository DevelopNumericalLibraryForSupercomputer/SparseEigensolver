#pragma once
#include <algorithm>
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"

#include "../device/TensorOp.hpp"
//#include "../device/mkl/TensorOp.hpp"


namespace SE{

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class TensorOperations{
public:
    TensorOperations(){};

    virtual std::unique_ptr<DenseTensor<1, DATATYPE, mtype, device> > matvec(const DenseTensor<1, DATATYPE, mtype, device>& vec) const=0;
    virtual std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > matvec(const DenseTensor<2, DATATYPE, mtype, device>& vec) const=0;
    virtual DATATYPE get_diag_element(const int index)const =0;
    virtual std::array<int, 2> get_global_shape() const=0;

};


template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class DenseTensorOperations: public TensorOperations<DATATYPE,mtype,device>{
    //default class
public:
    DenseTensorOperations(const DenseTensor<2, DATATYPE, mtype, device>* p_tensor):p_tensor(p_tensor){};

    std::unique_ptr<DenseTensor<1, DATATYPE, mtype, device> > matvec(const DenseTensor<1, DATATYPE, mtype, device>& vec) const override{
        return TensorOp::matmul(*this->p_tensor, vec);
    };
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > matvec(const DenseTensor<2, DATATYPE, mtype, device>& vec) const override{
        return TensorOp::matmul(*this->p_tensor, vec);
    };
    DATATYPE get_diag_element(const int index) const override{
        std::array<int, 2> global_array_index = {index, index};
        auto local_index = p_tensor->ptr_map->global_to_local(p_tensor->ptr_map->unpack_global_array_index(global_array_index));
        DATATYPE buff[2] = {0.0, 0.0};
        if (local_index>=0){
            buff[0] = p_tensor->operator()(local_index);  
        }
        p_tensor->ptr_comm->allreduce(&buff[0], 1, &buff[1], OPTYPE::SUM);
        return buff[1];
    };

    std::array<int, 2> get_global_shape() const override{
        return p_tensor->ptr_map->get_global_shape();
    };

private:
    const DenseTensor<2, DATATYPE, mtype, device>* p_tensor;
};

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class SparseTensorOperations: public TensorOperations<DATATYPE,mtype,device>{
    //default class
public:
    SparseTensorOperations(const SparseTensor<2, DATATYPE, mtype, device>* p_tensor):p_tensor(p_tensor){};

    std::unique_ptr<DenseTensor<1, DATATYPE, mtype, device> > matvec(const DenseTensor<1, DATATYPE, mtype, device>& vec) const override{
        return TensorOp::matmul(*this->p_tensor, vec);
    };
    std::unique_ptr<DenseTensor<2, DATATYPE, mtype, device> > matvec(const DenseTensor<2, DATATYPE, mtype, device>& vec) const override{
        return TensorOp::matmul(*this->p_tensor, vec);
    };
    DATATYPE get_diag_element(const int index) const override{
        DATATYPE buff[2] = {0.0,0.0};
        std::array<int, 2> global_array_index = {index, index};
        auto rank  =  p_tensor->ptr_map->find_rank_from_global_array_index(global_array_index);
        if(rank == p_tensor->ptr_comm->get_rank()){
            auto pos = std::find_if(p_tensor->data.begin(), p_tensor->data.end(), [global_array_index](const std::pair<std::array<int,2>, DATATYPE>& element) { return element.first == global_array_index; });

            if(pos!= p_tensor->data.end()) {
                buff[0] = (*pos).second;
            }

        }
        p_tensor->ptr_comm->allreduce(&buff[0], 1, &buff[1], OPTYPE::SUM);
        return buff[1];
    };

    std::array<int, 2> get_global_shape() const override{
        return p_tensor->ptr_map->get_global_shape();
    };

private:
    const SparseTensor<2, DATATYPE, mtype, device>* p_tensor;
};

}
