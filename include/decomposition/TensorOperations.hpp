#pragma once
#include "../DenseTensor.hpp"
#include "../SparseTensor.hpp"
#include "../Contiguous1DMap.hpp"
#include "../Device.hpp"

#include "../device/TensorOp.hpp"
#include "../device/mkl/TensorOp.hpp"


namespace SE{

class TensorOperations{
public:
    TensorOperations(){};

    virtual DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> matvec(const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec)=0;
    virtual DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> matvec(const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& vec)=0;
    virtual double get_diag_element(const int index)=0;
    virtual std::array<int, 2> get_global_shape()=0;

};

class DenseTensorOperations: public TensorOperations{
    //default class
public:
    DenseTensorOperations(DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& tensor):tensor(tensor){};

    DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> matvec(const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> matvec(const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };
    double get_diag_element(const int index) override{
        std::array<int, 2> array_index = {index, index};
        return tensor.operator()(tensor.map.global_to_local(tensor.map.unpack_global_array_index(array_index)));
    };

    std::array<int, 2> get_global_shape() override{
        return tensor.map.get_global_shape();
    };

private:
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> tensor;
};

class SparseTensorOperations: public TensorOperations{
    //default class
public:
    SparseTensorOperations(SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& tensor):tensor(tensor){};

    DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> matvec(const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> matvec(const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& vec) override{
        return TensorOp::matmul(this->tensor, vec);
    };

    double get_diag_element(const int index) override{
        std::cout << "SPARSETENSOR does not have get method." << std::endl;
        exit(-1);
    };

    std::array<int, 2> get_global_shape() override{
        return tensor.map.get_global_shape();
    };

private:
    SparseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> tensor;
};

}