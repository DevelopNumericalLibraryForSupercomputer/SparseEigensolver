#https://gitlab.com/jhwoo15/tucy/-/blob/master/python/array.pxd?ref_type=heads

# cython: language_level=3
# distutils: language = c++
cdef extern from *:
    ctypedef size_t one   "1" 
    ctypedef size_t two   "2" 
    ctypedef size_t three "3" 
    ctypedef size_t four  "4" 
    ctypedef size_t five  "5" 
    ctypedef size_t six   "6" 

cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass array[datatype, dimension]:
      array() except+
      datatype& operator[](size_t)

#사용예시
#from array cimport array, six
#    cdef:
#        array[size_t, six] c_shape
#        array[size_t, six] c_ranks