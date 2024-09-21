#pragma once
#include "decomposition/TensorOperations.hpp"

namespace SE{

template<typename DATATYPE, MTYPE mtype, DEVICETYPE device>
class TestTensorOperations: public TensorOperations<DATATYPE,mtype, device>{
public:
    TestTensorOperations(){};
    TestTensorOperations(int matrix_size):matrix_size(matrix_size){};
// n by n kinetic energy matrix-like matrix generator
    std::unique_ptr<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> >matvec(const DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) const override{
        auto return_vec = std::make_unique<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> >(vec);
        //const int n = vec.map.get_global_shape()[0];
        DATATYPE invh2 = 1.0;
        for(int i=0;i<matrix_size;i++){
            return_vec->data[i] = vec.data[i]* ( 2.0*((DATATYPE)i-(DATATYPE)matrix_size)   - invh2*5.0/2.0 );
            /*
            if(i%17==0){
                for(int j=0;j<matrix_size;j=j+17){
                    if(i!=j) return_vec->data[i] += vec.data[j]*0.01;
                }
            }
            */
        }
        for(int i=1;i<matrix_size;i++){
            return_vec->data[i] += vec.data[i-1]* ( invh2*4.0/3.0);
        }
        for(int i=0;i<matrix_size-1;i++){
            return_vec->data[i] += vec.data[i+1]* ( invh2*4.0/3.0);
        }
        for(int i=2;i<matrix_size;i++){
            return_vec->data[i] += vec.data[i-2]* (invh2*(-1.0)/12.0);
        }
        for(int i=0;i<matrix_size-2;i++){
            return_vec->data[i] += vec.data[i+2]* (invh2*(-1.0)/12.0);
        }
        
        for(int i=3;i<matrix_size;i++){
            return_vec->data[i] += vec.data[i-3]* 0.3;
        }
        for(int i=0;i<matrix_size-3;i++){
            return_vec->data[i] += vec.data[i+3]* 0.3;
        }

        for(int i=4;i<matrix_size;i++){
            return_vec->data[i] += vec.data[i-4]* (-0.1);
        }
        for(int i=0;i<matrix_size-4;i++){
            return_vec->data[i] += vec.data[i+4]* (-0.1);
        }
        return return_vec;
    };

    std::unique_ptr<DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> > matvec(const DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>& vec) const override{
        const int num_vec = vec.ptr_map->get_global_shape()[1];
        //const int matrix_size = vec.map.get_global_shape()[0];

        auto return_vec = std::make_unique<DenseTensor<2, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL> > (vec);

        std::array<int, 1> oned_shape = {matrix_size};
		Contiguous1DMapInp<1> vec_map_inp(oned_shape);
		std::unique_ptr< Map<1,MTYPE::Contiguous1D> > ptr_vec_map = vec_map_inp.create_map();

        //std::array<DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>, num_vec> vec_array;
        for(int i=0;i<num_vec;i++){
            auto tmp_vec = DenseTensor<1, DATATYPE, MTYPE::Contiguous1D, DEVICETYPE::MKL>(vec.copy_comm(),ptr_vec_map);
            copy<DATATYPE, DEVICETYPE::MKL>(matrix_size,&vec.data[i],num_vec,tmp_vec.data.get(),1);
            auto tmp_result = matvec(tmp_vec);
            copy<DATATYPE, DEVICETYPE::MKL>(matrix_size,tmp_result->data.get(),1,&return_vec->data.get()[i],num_vec);
        }
        return return_vec;
    };
    DATATYPE get_diag_element(const int index) const override{
        return 2.0*((DATATYPE)index-(DATATYPE)matrix_size)   - 5.0/2.0;
    };

    std::array<int, 2> get_global_shape() const override{
        std::array<int, 2> return_array = {matrix_size,matrix_size};
        return return_array;
    };

private:
    int matrix_size;
};

}
