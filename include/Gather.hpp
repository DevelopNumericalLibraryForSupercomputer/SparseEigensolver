#pragma once 
#include <algorithm>
#include "Map.hpp"
#include "Contiguous1DMap.hpp"
#include "Comm.hpp"
#include "Type.hpp"
#include "DenseTensor.hpp"

namespace SE{

template<typename MAPTYPE> 
class Gather{
public:
template<typename DATATYPE, DEVICETYPE device>
static void gather_from_all(DATATYPE* src, const MAPTYPE& map, const Comm<device>& comm, DATATYPE* trg);

template<typename DATATYPE, DEVICETYPE device>
static void gather_from_all(DATATYPE* src, DenseTensor<1,DATATYPE,MAPTYPE,device>& output)
{
    gather_from_all(src, output.map, output.comm, output.data );
    return;
};

};



template<>
template<typename DATATYPE, DEVICETYPE device>
void Gather<Contiguous1DMap<1>>::gather_from_all(DATATYPE* src, const Contiguous1DMap<1>& map, const Comm<device>& comm, DATATYPE* trg){
    size_t start_idx=0;
    auto all_local_shape = map.get_all_local_shape();
    std::for_each( all_local_shape.begin(), all_local_shape.begin()+comm.get_rank(),
                               [&start_idx](const std::array<size_t,1>& array) {start_idx+=array[0];} );
    memcpy<DATATYPE,device>(trg, src+start_idx, all_local_shape[comm.get_rank()][0]  );

    return;
}


}
