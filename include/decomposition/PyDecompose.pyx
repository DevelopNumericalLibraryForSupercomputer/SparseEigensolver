#distutils: language = c++
# PyDecompose.pyx
# Decompose.hpp의 Cython wrapper

import numpy as np
cimport numpy as np
from cpython cimport PyObject, Py_INCREF, Py_DECREF
from libcpp cimport size_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
np.import_array()
#PyOperations.pyx에서 정의한 Tensor_Operations 클래스를 import
from PyOperations cimport Tensor_Operations

#../device/mkl/MKLComm.hpp의 create_comm 함수를 import
cdef extern from "../device/mkl/MKLComm.hpp" namespace "SE":
    unique_ptr[Comm[DEVICETYPE::MKL]] create_comm(int argc, char *argv[])


#../DenseTensor.hpp의 DenseTensor 클래스를 import
cdef extern from "../DenseTensor.hpp" namespace "SE":
    cdef cppclass DenseTensor[N, DATATYPE, MAPTYPE, DEVICETYPE]:
        DenseTensor(const Comm[DEVICETYPE]& input_comm, const MAPTYPE input_map, DATATYPE* input_data)
        DATATYPE* copy_data()    

#DecomposeResult.hpp의 DecomposeResult 클래스를 import
cdef extern from "DecomposeResult.hpp" namespace "SE":
    cdef cppclass DecomposeResult[DATATYPE]:
        DecomposeResult(size_t num_eig, unique_ptr[DATATYPE[]] real_eigvals, unique_ptr[DATATYPE[]] imag_eigvals)

# Declare the external C++ function
cdef extern from "Decompose.hpp" namespace "SE":
    unique_ptr[DecomposeResult[DATATYPE]] davidson[DATATYPE, MAPTYPE, DEVICETYPE](TensorOperations* operations, DenseTensor[size_t, DATATYPE, MAPTYPE, DEVICETYPE]* eigvec, string method)

def c_decompose(Tensor_Operations py_operations, np.ndarray eigvec, string method):
    # Convert the numpy ndarray to a pointer using memoryview
    cdef double* eigvec_ptr = <double*> eigvec.data
    # Create a DenseTensor object


    return davidson[double, MAPTYPE, DEVICETYPE](<TensorOperations*> &py_operations._operations, eigvec, method)