# cython: language_level=3
# distutils: language = c++

from array cimport array, one, two, three, four, five, six
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Map.hpp" namespace "SE":
    cdef cppclass Map[dimension, mtype]:
        pass
        #Comm() except+
        #Comm(int rank, int world_size) except+
        #Comm[DEVICETYPE]* clone()


cdef extern from "../include/Contiguous1DMap.hpp" namespace "SE":
    cppclass Contiguous1DMap[dimension]:
        pass
cdef extern from "../include/BlockCyclingMap.hpp" namespace "SE":
    cppclass BlockCyclingMap[dimension]:
        pass
