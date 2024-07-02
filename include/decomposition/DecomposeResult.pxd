

# decompose_result.pyx
cimport decompose_result
from libcpp.memory cimport unique_ptr
import numpy as np

cdef class PyDecomposeResult:
    cdef unique_ptr[decompose_result.DecomposeResult[double]] _result

    def __cinit__(self, size_t num_eig, real_eigvals, imag_eigvals):
        cdef unique_ptr[double[]] c_real_eigvals = unique_ptr[double[]](new double[num_eig])
        cdef unique_ptr[double[]] c_imag_eigvals = unique_ptr[double[]](new double[num_eig])
        
        for i in range(num_eig):
            c_real_eigvals[i] = real_eigvals[i]
            c_imag_eigvals[i] = imag_eigvals[i]

        self._result = unique_ptr[decompose_result.DecomposeResult[double]](
            new decompose_result.DecomposeResult[double](num_eig, move(c_real_eigvals), move(c_imag_eigvals))
        )

    @property
    def num_eig(self):
        return self._result.get().num_eig

    @property
    def real_eigvals(self):
        return np.asarray(<double[:self._result.get().num_eig]>self._result.get().real_eigvals.get())

    @property
    def imag_eigvals(self):
        return np.asarray(<double[:self._result.get().num_eig]>self._result.get().imag_eigvals.get())