# cython: language_level=3
# distutils: language = c++

from array cimport array, one, two, three, four, five, six
from libcpp.vector cimport vector
from libcpp cimport bool


cdef extern from "../include/Contiguous1DMap.hpp" namespace "SE":
    cppclass Contiguous1DMap[dimension]:
        Contiguous1DMap() except+
        Contiguous1DMap(const array[int, dimension] global_shape, const int my_rank, const int world_size) except+
        Contiguous1DMap(const array[int, dimension] global_shape, const int my_rank, const int world_size, const array[bool, dimension] is_parallel ) except+
        Contiguous1DMap(const array[int, dimension] global_shape, const int my_rank, const int world_size, const array[int, dimension] ranks_per_dim ) except+

        Contiguous1DMap[dimension]* clone()
    
        int get_num_local_elements()
    
        # global index <-> local index
        int local_to_global(int local_index) 
        int global_to_local(int global_index) 
        array[int, dimension] local_to_global(array[int, dimension] local_array_index) 
        array[int, dimension] global_to_local(array[int, dimension] global_array_index) 
                                                                                
        # local array index <-> local index 
        int  unpack_local_array_index(array[int, dimension] local_array_index) 
        array[int, dimension] pack_local_index(int local_index)
        int  unpack_global_array_index(array[int, dimension] global_array_index) 
        array[int, dimension] pack_global_index(int global_index) 

        int find_rank_from_global_index(int global_index) 
        int find_rank_from_global_array_index(array[int, dimension] global_array_index) 

        #vector[array[int, dimension]] get_all_local_shape() 
        int get_split_dim() 

        vector[array[int, dimension]] all_local_shape
        int split_dim
        initialize()