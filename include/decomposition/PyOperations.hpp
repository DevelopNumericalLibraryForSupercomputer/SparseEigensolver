// PyTensorOperations.hpp
// wrapper class of Matrix vector product callback function pointer for cython interface
#pragma once
#include "TensorOperations.hpp"

namespace SE{

typedef void   (*MatrixOneVecCallback)(const double* input_vec, double* output_vec, const size_t size, void* user_data);
typedef void   (*MatrixMultVecCallback)(const double* input_vecs, double* output_vec, const size_t num_vec, const size_t size, void* user_data);
typedef double (*GetDiagElementCallback)(size_t index, void* user_data);
typedef void   (*GetGlobalShapeCallback)(size_t* shape, void* user_data);

class PyTensorOperations: public TensorOperations{
public:
    MatrixOneVecCallback matonevec_callback;
    MatrixMultVecCallback matmultvec_callback;
    GetDiagElementCallback getdiag_callback;
    GetGlobalShapeCallback getshape_callback;
    void* user_data;

    PyTensorOperations(){};
    PyTensorOperations(MatrixOneVecCallback mov, MatrixMultVecCallback mmv, GetDiagElementCallback gde, GetGlobalShapeCallback ggs, void* data):
         matonevec_callback(mov), matmultvec_callback(mmv), getdiag_callback(gde), getshape_callback(ggs), user_data(data){};
    
    DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> matvec(const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec) override;
    DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> matvec(const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& vec) override;
    double get_diag_element(const size_t index) override;
    std::array<size_t, 2> get_global_shape() override;
};

DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL> PyTensorOperations::matvec(const DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>& vec){
    auto return_vec = DenseTensor<1, double, Contiguous1DMap<1>, DEVICETYPE::MKL>(vec);
    size_t size = vec.map.get_global_shape(0);
    double* input_vec = vec.copy_data();
    double* output_vec = new double[size];
    matonevec_callback(input_vec, output_vec, size, this->user_data);

    return return_vec;
};

DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL> PyTensorOperations::matvec(const DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>& vec){
    auto return_vec = DenseTensor<2, double, Contiguous1DMap<2>, DEVICETYPE::MKL>(vec);
    size_t size = vec.map.get_global_shape(0);
    size_t num_vec = vec.map.get_global_shape(1);
    double* input_vec = vec.copy_data();
    double* output_vec = new double[num_vec*size];
    matmultvec_callback(input_vec, output_vec, num_vec, size, this->user_data);

    return return_vec;
};

double PyTensorOperations::get_diag_element(const size_t index){
    return getdiag_callback(index, this->user_data);
};

std::array<size_t, 2> PyTensorOperations::get_global_shape(){
    size_t shape[2];
    getshape_callback(shape, this->user_data);
    return {shape[0], shape[1]};
};

} 

