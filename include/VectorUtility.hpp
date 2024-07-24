#include "Type.hpp"
#include "device/LinearOp.hpp"

namespace SE{
template <typename DATATYPE, DEVICETYPE device>
void eigenvec_sort(DATATYPE* eigvals, DATATYPE* eigvecs, const int number_of_eigvals, const int vector_size){
    DATATYPE* new_eigvals = new DATATYPE[number_of_eigvals];
    DATATYPE* new_eigvecs = new DATATYPE[number_of_eigvals*vector_size];
    std::vector<int> sorted_indicies = sort_indicies<DATATYPE>(eigvals, number_of_eigvals);
    for(int i=0;i<number_of_eigvals;i++){
        new_eigvals[i] = eigvals[sorted_indicies[i]];
        for(int j=0;j<vector_size;j++){
            new_eigvecs[i*number_of_eigvals+j] = eigvecs[sorted_indicies[i]*number_of_eigvals+j];
        }
    }
    
    memcpy<DATATYPE, device>(eigvals, new_eigvals, number_of_eigvals, COPYTYPE::DEVICE2DEVICE);
    memcpy<DATATYPE, device>(eigvecs, new_eigvecs, number_of_eigvals*vector_size, COPYTYPE::DEVICE2DEVICE);
}

}