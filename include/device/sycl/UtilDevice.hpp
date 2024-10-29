#pragma once
#include <cassert>
#include "../../Utility.hpp"
#include "../LinearOp.hpp"
#include "oneapi/mkl.hpp"

//#include "mkl.h"
namespace SE{

oneapi::mkl::job map_jobz_SYCL(char jobz){
    switch(jobz){
        case 'N': return oneapi::mkl::job::novec;
        case 'V': return oneapi::mkl::job::vec;
    }    
    assert (false);
}

oneapi::mkl::uplo map_uplo_SYCL(char uplo){
    switch (uplo){
        case 'U': return oneapi::mkl::uplo::upper;
        case 'L': return oneapi::mkl::uplo::lower;
    }
    exit(-1);
}

oneapi::mkl::transpose map_transpose_SYCL(TRANSTYPE trans){
    switch (trans){
        case TRANSTYPE::N:      return oneapi::mkl::transpose::nontrans;
        case TRANSTYPE::T:      return oneapi::mkl::transpose::trans ;
        case TRANSTYPE::C:      return oneapi::mkl::transpose::conjtrans;
    }
    exit(-1);
}

}
