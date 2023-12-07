#pragma once 

namespace SE{

enum class DEVICETYPE: int{
    BASE=0, 
    MKL=1,
    MPI=2,
// accelerator that has independent memory space have larger than 10  
    CUDA=11,
    NCCL=12,

};

enum class MTYPE: int{
    Contiguous1D=0,
};


enum class COPYTYPE: int {
    NONE=0,
    HOST2DEVICE=1,
    DEVICE2HOST=2,
    DEVICE2DEVICE=3,
};

enum class STORETYPE:int{//data store type
    DENSE=0,
    COO=1,
};

enum class VERBOSETENSOR: int{
    SIMPLE=0,       // simple
    DEBUG1=1,       // detail
    DEBUG10=10,     // super detail
    DEBUG100=100,   // super super detail
};

enum class TRANSTYPE{
    N,
    T,
    C,
};

enum class ORDERTYPE{
    ROW,
    COL
};

enum class OPTYPE{//opertor for allreduce
    MAX,
    MIN,
    SUM,
    PROD
};

}

template<typename T>
struct Zero{
    static constexpr T value= (T) 0.0;
};

