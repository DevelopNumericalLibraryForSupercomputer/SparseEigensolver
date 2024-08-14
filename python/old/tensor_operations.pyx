#distutils: language = c++
# tensor_operations.pyx
import numpy as np
cimport numpy as np
from cython.operator cimport dereference
np.import_array()

from TensorOperations cimport TensorOperations
from array cimport array, two
from DenseTensor cimport DenseTensor, DenseTensor1D, DenseTensor2D

cdef class EOMCCSD_TensorOperations(TensorOperations):
    #미완성
        cdef TensorOperations* _tensor_ops

    def __cinit__(self):
        self._tensor_ops = new TensorOperations()

    def __dealloc__(self):
        del self._tensor_ops

    cpdef EOMCCSD_TensorOperations():
        pass

        DenseTensor1D matvec(const DenseTensor1D& vec)
        DenseTensor2D matvec(const DenseTensor2D& vec)
    
    cpdef double get_diag_element(self, const size_t index):
        pass

    cpdef array[size_t,two] get_global_shape(self):
        pass

    

    cdef TensorOperations* _tensor_ops

    def __cinit__(self):
        self._tensor_ops = new TensorOperations()

    def __dealloc__(self):
        del self._tensor_ops

    def matvec(self, np.ndarray[np.float64_t] vec):
        # Call appropriate matvec based on the dimension of vec
        
        if vec.ndim == 1:
            return self._tensor_ops.matvec(DenseTensor1D(vec)))
        elif vec.ndim == 2:
            return self._tensor_ops.matvec(DenseTensor1D(vec))
        else:
            raise ValueError("Unsupported dimension for vec")

    def get_diag_element(self, index):
        return self._tensor_ops.get_diag_element(index)

    def get_global_shape(self):
        return self._tensor_ops.get_global_shape()
