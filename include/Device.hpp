// Device information
// Read https://www.cppstories.com/2018/08/init-string-member/
#pragma once
//#include <string>
namespace SE{

enum class computEnv{
    MKL, CUDA,
    MPI, NCCL,
    //ROCM
};
/*
struct Device{
    const PROTOCOL protocol;
    constexpr Device(PROTOCOL info): protocol(info){};
};

struct Serial: public Device{
    constexpr Serial() : Device(SERIAL){};
};
struct MPI: public Device{
    constexpr MPI() : Device(MPI){};
};
struct CUDA: public Device{
    constexpr CUDA() : Device(NCCL){};
};
struct ROCm: public Device{
    constexpr ROCm() : Device(ROCM){};
};
*/
}


