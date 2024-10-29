#pragma once
#include <memory>
#include <vector>
#include "Utility.hpp"
#include "device/LinearOp.hpp"
namespace SE{

template <typename DATATYPE, DEVICETYPE device>
class DecomposeResult{
public:
    DecomposeResult(const int num_eig, 
                    DATATYPE* real_eigvals,
                    DATATYPE* imag_eigvals
                   ): num_eig(num_eig)
                   {
                     std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)> > uniq_real_eigvals(malloc<DATATYPE, device> (num_eig), free<device>) ; 
                     std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)> > uniq_imag_eigvals( malloc<DATATYPE, device> (num_eig), free<device>) ; 

                     memcpy<DATATYPE,device>(uniq_real_eigvals.get(), real_eigvals, num_eig, COPYTYPE::DEVICE2DEVICE);
                     memcpy<DATATYPE,device>(uniq_imag_eigvals.get(), imag_eigvals, num_eig, COPYTYPE::DEVICE2DEVICE);

                     this->real_eigvals = std::move(uniq_real_eigvals);
                     this->imag_eigvals = std::move(uniq_imag_eigvals);
                   };

    const int num_eig=0;
    std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)> > real_eigvals;
    std::unique_ptr<DATATYPE[],std::function<void(DATATYPE*)> > imag_eigvals;
};
/*
    DecomposeResult(const int num_eig, std::vector<typename real_type<DATATYPE>::type> real_eigvals, std::vector<typename real_type<DATATYPE>::type> imag_eigvals):
                     num_eig(num_eig), real_eigvals(real_eigvals), imag_eigvals(imag_eigvals){};
    const int num_eig=0;
    std::vector<typename real_type<DATATYPE>::type> real_eigvals;
    std::vector<typename real_type<DATATYPE>::type> imag_eigvals;
};
*/

}
