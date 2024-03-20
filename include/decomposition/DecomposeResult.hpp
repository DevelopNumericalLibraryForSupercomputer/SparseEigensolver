#pragma once
#include <memory>
#include <vector>
namespace SE{

template <typename DATATYPE>
class DecomposeResult{
public:
/*
    DecomposeResult(const size_t num_eig, 
                    std::unique_ptr< DATATYPE[] > real_eigvals,
                    std::unique_ptr< DATATYPE[] > imag_eigvals
                   ): num_eig(num_eig),
                     real_eigvals(std::move(real_eigvals )),
                     imag_eigvals(std::move(imag_eigvals ))
                   {};
    const size_t num_eig=0;
    std::unique_ptr<DATATYPE[] > real_eigvals;
    std::unique_ptr<DATATYPE[] > imag_eigvals;
};
*/
    DecomposeResult(const size_t num_eig, std::vector<DATATYPE> real_eigvals, std::vector<DATATYPE> imag_eigvals):
                     num_eig(num_eig), real_eigvals(real_eigvals), imag_eigvals(imag_eigvals){};
    const size_t num_eig=0;
    std::vector<DATATYPE> real_eigvals;
    std::vector<DATATYPE> imag_eigvals;
};

}
