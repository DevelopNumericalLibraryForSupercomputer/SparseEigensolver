#pragma once
#include "TensorOperations.hpp"

namespace SE{

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class TestTensorOperations: public TensorOperations<DATATYPE,mtype, device>{
public:
    TestTensorOperations(){};
    TestTensorOperations(int n):n(n){};
// n by n kinetic energy matrix-like matrix generator
    std::unique_ptr<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> >matvec(const DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) const override{
        auto return_vec = std::make_unique<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(vec);
        //const int n = vec.map.get_global_shape()[0];
        DATATYPE invh2 = 1.0;
        for(int i=0;i<n;i++){
            return_vec->data[i] = vec.data[i]* ( 2.0*((DATATYPE)i-(DATATYPE)n)   - invh2*5.0/2.0 );
        }
        for(int i=1;i<n;i++){
            return_vec->data[i] += vec.data[i-1]* ( invh2*4.0/3.0);
        }
        for(int i=0;i<n-1;i++){
            return_vec->data[i] += vec.data[i+1]* ( invh2*4.0/3.0);
        }
        for(int i=2;i<n;i++){
            return_vec->data[i] += vec.data[i-2]* (invh2*(-1.0)/12.0);
        }
        for(int i=0;i<n-2;i++){
            return_vec->data[i] += vec.data[i+2]* (invh2*(-1.0)/12.0);
        }

        return return_vec;
    };

    std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> > matvec(const DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) const override{
        const int num_vec = vec.ptr_map->get_global_shape()[1];
        //const int n = vec.map.get_global_shape()[0];

        auto return_vec = std::make_unique<DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (vec);

        std::array<int, 1> oned_shape = {n};
		Contiguous1DMapInp<1> vec_map_inp(oned_shape);
		std::unique_ptr< Map<1,MTYPE::Contiguous1D> > ptr_vec_map = vec_map_inp.create_map();

        //std::array<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>, num_vec> vec_array;
        for(int i=0;i<num_vec;i++){
            auto tmp_vec = DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>(vec.copy_comm(),ptr_vec_map);
            copy<DATATYPE, DEVICETYPE::MKL>(n,&vec.data[i],num_vec,tmp_vec.data.get(),1);
            auto tmp_result = matvec(tmp_vec);
            copy<DATATYPE, DEVICETYPE::MKL>(n,tmp_result->data.get(),1,&return_vec->data.get()[i],num_vec);
        }
        return return_vec;
    };
    DATATYPE get_diag_element(const int index) const override{
        return 2.0*((DATATYPE)index-(DATATYPE)n)   - 5.0/2.0;
    };

    std::array<int, 2> get_global_shape() const override{
        std::array<int, 2> return_array = {n,n};
        return return_array;
    };

    private:
    int n;
};

}
