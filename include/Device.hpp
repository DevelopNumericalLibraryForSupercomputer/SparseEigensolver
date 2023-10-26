// Device information
// Read https://www.cppstories.com/2018/08/init-string-member/
#pragma once
#include <string>
//#include <iostream>
namespace SE{

enum class computEnv: int {
    BASE=0,
    MKL =1, 
    CUDA=2,
    MPI =3, 
    NCCL=4,
    //ROCM
};

std::ostream& operator<< (std::ostream& os, computEnv comput_env)
{
    switch (comput_env)
    {
        case computEnv::BASE: return os << "BASE";
        case computEnv::MKL : return os << "MKL" ;
        case computEnv::CUDA: return os << "CUDA";
        case computEnv::MPI : return os << "MPI";
        case computEnv::NCCL: return os << "NCCL";
        // omit default case to trigger compiler warning for missing cases
    };
    return os;
    //return os << static_cast<std::uint16_t>(ethertype);
}

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


