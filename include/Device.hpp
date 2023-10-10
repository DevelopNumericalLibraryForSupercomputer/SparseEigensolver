// Device information
// Read https://www.cppstories.com/2018/08/init-string-member/
#pragma once
#include <string>
namespace SE{

typedef enum{//opertor for allreduce
    SE_SERIAL,
    SE_MPI,
    SE_NCCL,
    SE_ROCM
} SE_PROTOCOL;


struct Device{
    const SE_PROTOCOL protocol;
    constexpr Device(SE_PROTOCOL info): protocol(info){};
};

struct Serial: public Device{
    constexpr Serial() : Device(SE_SERIAL){};
};
struct MPI: public Device{
    constexpr MPI() : Device(SE_MPI){};
};
struct CUDA: public Device{
    constexpr CUDA() : Device(SE_NCCL){};
};
struct ROCm: public Device{
    constexpr ROCm() : Device(SE_ROCM){};
};

}


