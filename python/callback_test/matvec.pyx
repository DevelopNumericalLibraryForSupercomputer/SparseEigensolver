import numpy as np
cimport numpy as np

def matvec(np.ndarray[double, ndim=2] arr):
    cdef np.ndarray[double, ndim=2] result = arr * np.arange(1, arr.shape[1] + 1)
    return result

#cdef public void c_matvec_wrapper(double* arr_ptr, int rows, int cols, double* result_ptr):
#    cdef np.ndarray[double, ndim=2] arr = np.asarray(<double[:rows, :cols]> arr_ptr)
#    cdef np.ndarray[double, ndim=2] result = matvec(arr)
#    cdef np.ndarray[double, ndim=2] result_view = np.asarray(<double[:rows, :cols]> result_ptr)
#    result_view[:] = result

# Python-callable wrapper
#def py_c_matvec_wrapper(np.ndarray[double, ndim=2] arr):
#    cdef np.ndarray[double, ndim=2] result = np.zeros_like(arr)
#    c_matvec_wrapper(&arr[0,0], arr.shape[0], arr.shape[1], &result[0,0])
#    return result