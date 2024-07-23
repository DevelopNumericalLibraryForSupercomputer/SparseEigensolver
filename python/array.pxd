#https://gitlab.com/jhwoo15/tucy/-/blob/master/python/array.pxd?ref_type=heads

# cython: language_level=3
# distutils: language = c++
cdef extern from *:
    ctypedef int one   "1" 
    ctypedef int two   "2" 
    ctypedef int three "3" 
    ctypedef int four  "4" 
    ctypedef int five  "5" 
    ctypedef int six   "6" 

cdef extern from "<array>" namespace "std" nogil:
    cdef cppclass array[datatype, dimension]:
      array() except+
      datatype& operator[](int)

#사용예시
#from array cimport array, six
#    cdef:
#        array[int, six] c_shape
#        array[int, six] c_ranks