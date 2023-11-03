// Comput Environment information
#pragma once
#include <string>
//#include <iostream>
namespace SE{
/*
enum class computEnv{
    BASE,
    MKL, 
    CUDA,
    MPI, 
    NCCL,
    //ROCM
};
*/
/*
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
*/

//std::ostream& operator<< (std::ostream& os, computEnv comput_env){ return os << comput_env.env_name;};

struct ComputEnv{

    const std::string env_name;
    constexpr ComputEnv(): env_name("BASE"){};
    //const computEnv env = SE::BASE;
    //constexpr ComputEnv(computEnv env_): env(env_)){};
};

struct MKL: public ComputEnv{
    const std::string env_name = "MKL";
    //constexpr MKL() : ComputEnv(SE::MKL){};
};
struct MPI: public ComputEnv{
    const std::string env_name = "MPI";
    //constexpr MPI() : ComputEnv(SE::MPI){};
};
struct CUDA: public ComputEnv{
    const std::string env_name = "CUDA";
    //constexpr CUDA() : ComputEnv(SE::NCCL){};
};
struct ROCm: public ComputEnv{
    const std::string env_name = "ROCm";
    //constexpr ROCm() : ComputEnv(SE::ROCM){};
};

}


