#pragma once
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"

#include "../device/TensorOp.hpp"
//#include "../device/mkl/TensorOp.hpp"


namespace SE{

template<MTYPE mtype, DEVICETYPE device>
class TensorOperations{
public:
    TensorOperations(){};

    virtual DenseTensor<1, double, mtype, device> matvec(const DenseTensor<1, double, mtype, device>& vec) const=0;
    virtual DenseTensor<2, double, mtype, device> matvec(const DenseTensor<2, double, mtype, device>& vec) const=0;
    virtual double get_diag_element(const int index)const =0;
    virtual std::array<int, 2> get_global_shape() const=0;

};


template<MTYPE mtype, DEVICETYPE device>
class DenseTensorOperations: public TensorOperations<mtype,device>{
    //default class
public:
    DenseTensorOperations(DenseTensor<2, double, mtype, device>& tensor):tensor(tensor){};

    DenseTensor<1, double, mtype, device> matvec(const DenseTensor<1, double, mtype, device>& vec) const override{
        return TensorOp::matmul(this->tensor, vec);
    };
    DenseTensor<2, double, mtype, device> matvec(const DenseTensor<2, double, mtype, device>& vec) const override{
        return TensorOp::matmul(this->tensor, vec);
    };
    double get_diag_element(const int index) const override{
        std::array<int, 2> global_array_index = {index, index};
		auto local_index = tensor.ptr_map->global_to_local(tensor.ptr_map->unpack_global_array_index(global_array_index));
		double src = 0.0; double trg = 0.0;

		if (local_index>=0){
			src = tensor(local_index);	
		}
		tensor.ptr_comm->allreduce(&src, 1, &trg, OPTYPE::SUM);
        return trg;
    };

    std::array<int, 2> get_global_shape() const override{
        return tensor.ptr_map->get_global_shape();
    };

private:
    DenseTensor<2, double, mtype, device> tensor;
};

template<MTYPE mtype, DEVICETYPE device>
class SparseTensorOperations: public TensorOperations<mtype,device>{
    //default class
public:
    SparseTensorOperations(SparseTensor<2, double, mtype, device>& tensor):tensor(tensor){};

    DenseTensor<1, double, mtype, device> matvec(const DenseTensor<1, double, mtype, device>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };
    DenseTensor<2, double, mtype, device> matvec(const DenseTensor<2, double, mtype, device>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };

    double get_diag_element(const int index) override{
        std::cout << "SPARSETENSOR does not have get method." << std::endl;
        exit(-1);
    };

    std::array<int, 2> get_global_shape() override{
        return tensor.ptr_map->get_global_shape();
    };

private:
    SparseTensor<2, double, mtype, device> tensor;
};

}
