#pragma once
#include "../../Utility.hpp"
#include "mkl.h"

namespace SE{
CBLAS_LAYOUT map_layout_blas_MKL(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return CblasRowMajor;
        case SE_layout::ColMajor: return CblasColMajor;
    }
    exit(-1);
}

int map_layout_lapack_MKL(SE_layout layout){
    switch (layout){
        case SE_layout::RowMajor: return LAPACK_ROW_MAJOR;
        case SE_layout::ColMajor: return LAPACK_COL_MAJOR;
    }
    exit(-1);
}

CBLAS_TRANSPOSE map_transpose_blas_MKL(SE_transpose trans){
    switch (trans){
        case SE_transpose::NoTrans:   return CblasNoTrans;
        case SE_transpose::Trans:     return CblasTrans;
        case SE_transpose::ConjTrans: return CblasConjTrans;
    }
    exit(-1);
}


}
