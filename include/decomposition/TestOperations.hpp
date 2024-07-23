#pragma once
#include "TensorOperations.hpp"

namespace SE{

template<MTYPE mtype, DEVICETYPE device>
class TestTensorOperations: public TensorOperations<mtype, device>{
public:
    TestTensorOperations(){};
    TestTensorOperations(int n):n(n){};
// n by n kinetic energy matrix-like matrix generator
    DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> matvec(const DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) override{
        auto return_vec = DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(vec);
        //const int n = vec.map.get_global_shape()[0];
        double invh2 = 1.0;
        for(int i=0;i<n;i++){
            return_vec.data[i] = vec.data[i]* ( 2.0*((double)i-(double)n)   - invh2*5.0/2.0 );
        }
        for(int i=1;i<n;i++){
            return_vec.data[i] += vec.data[i-1]* ( invh2*4.0/3.0);
        }
        for(int i=0;i<n-1;i++){
            return_vec.data[i] += vec.data[i+1]* ( invh2*4.0/3.0);
        }
        return return_vec;
    };

    DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL> matvec(const DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) override{
        const int num_vec = vec.ptr_map->get_global_shape()[1];
        //const int n = vec.map.get_global_shape()[0];

        auto return_vec = DenseTensor<2, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(vec);

        std::array<int, 1> oned_shape = {n};
		Contiguous1DMapInp<1> vec_map_inp(oned_shape);
		std::unique_ptr< Map<1,MTYPE::Contiguous1D> > ptr_vec_map = vec_map_inp.create_map();

        //std::array<DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>, num_vec> vec_array;
        for(int i=0;i<num_vec;i++){
            auto tmp_vec = DenseTensor<1, double, MTYPE::Contiguous1D, DEVICETYPE::MKL>(vec.copy_comm(),ptr_vec_map);
            copy<double, DEVICETYPE::MKL>(n,&vec.data[i],num_vec,tmp_vec.data,1);
            auto tmp_result = matvec(tmp_vec);
            copy<double, DEVICETYPE::MKL>(n,tmp_result.data,1,&return_vec.data[i],num_vec);
        }
        return return_vec;
    };
    double get_diag_element(const int index) override{
        return 2.0*((double)index-(double)n)   - 5.0/2.0;
    };

    std::array<int, 2> get_global_shape() override{
        std::array<int, 2> return_array = {n,n};
        return return_array;
    };

    private:
    int n;
};

}
