#pragma once
#include "../../Utility.hpp"
#include "mkl.h"

namespace SE{

CBLAS_LAYOUT map_order_blas_MPI(ORDERTYPE layout){
    switch (layout){
        case ORDERTYPE::ROW: return CblasRowMajor;
        case ORDERTYPE::COL: return CblasColMajor;
    }
    exit(-1);
}

int map_order_lapack_MPI(ORDERTYPE layout){
    switch (layout){
        case ORDERTYPE::ROW: return LAPACK_ROW_MAJOR;
        case ORDERTYPE::COL: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose_blas_MPI(TRANSTYPE trans){
    switch (trans){
        case TRANSTYPE::N:      return CblasNoTrans;
        case TRANSTYPE::T:      return CblasTrans;
        case TRANSTYPE::C:      return CblasConjTrans;
    }
    exit(-1);
}

}
