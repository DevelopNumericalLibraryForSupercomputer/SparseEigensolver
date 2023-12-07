#pragma once
#include "../../Utility.hpp"
#include "mkl.h"

namespace SE{

CBLAS_LAYOUT map_layout_blas_MPI(SE_layout layout){
    switch (layout){
        case LAYOUT::ROW: return CblasRowMajor;
        case LAYOUT::COL: return CblasColMajor;
    }
    exit(-1);
}

int map_layout_lapack_MPI(SE_layout layout){
    switch (layout){
        case LAYOUT::ROW: return LAPACK_ROW_MAJOR;
        case LAYOUT::COL: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose_blas_MPI(SE_transpose trans){
    switch (trans){
        case TRANSPOSE::N:      return CblasNoTrans;
        case TRANSPOSE::T:      return CblasTrans;
        case TRANSPOSE::C:      return CblasConjTrans;
    }
    exit(-1);
}

}
