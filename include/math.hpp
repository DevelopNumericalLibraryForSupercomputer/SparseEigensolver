#pragma once 
#include "Device.hpp"

template <typename datatype, PROTOCOL protocol >
int geev();


#ifdef SE_SUPPORT_MKL
template<>
int geev<double,SERIAL>(){
    int info = LAPACKE_dgeev( LAPACK_COL_MAJOR, 'V', 'V', n, this->data, n, real_eigvals.get(), imag_eigvals.get(),
                              eigvec_0, n, eigvec_1, n );
    return info 
}
#endif 

#ifdef SE_SUPPORT_CUDA
#endif 
