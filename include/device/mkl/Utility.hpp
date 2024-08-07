#pragma once
#include "../../Utility.hpp"
#include "../LinearOp.hpp"
#include "mkl.h"

namespace SE{

CBLAS_LAYOUT map_order_blas_MKL(ORDERTYPE order){
    switch (order){
        case ORDERTYPE::ROW: return CblasRowMajor;
        case ORDERTYPE::COL: return CblasColMajor;
    }
    exit(-1);
}

CBLAS_UPLO map_uplo_blas_MKL(char uplo){
    switch (uplo){
        case 'U': return CblasUpper;
        case 'L': return CblasLower;
    }
    exit(-1);
}
int map_order_lapack_MKL(ORDERTYPE order){
    switch (order){
        case ORDERTYPE::ROW: return LAPACK_ROW_MAJOR;
        case ORDERTYPE::COL: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose_blas_MKL(TRANSTYPE trans){
    switch (trans){
        case TRANSTYPE::N:      return CblasNoTrans;
        case TRANSTYPE::T:      return CblasTrans;
        case TRANSTYPE::C:      return CblasConjTrans;
    }
    exit(-1);
}

char map_order_blas_extension_MKL(ORDERTYPE order){
    switch (order){
        case ORDERTYPE::ROW: return 'R';
        case ORDERTYPE::COL: return 'C';
    }
    exit(-1);
}

char map_transpose_blas_extension_MKL(TRANSTYPE trans){
    switch (trans){
        case TRANSTYPE::N:      return 'N';
        case TRANSTYPE::T:      return 'T';
        case TRANSTYPE::C:      return 'C';
    }
    exit(-1);
}

}
