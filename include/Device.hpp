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
    public:
    const char* env_name;
    constexpr ComputEnv(const char* _env_name="BASE"): env_name(_env_name){};
    
    //const computEnv env = SE::BASE;
    //constexpr ComputEnv(computEnv env_): env(env_)){};
};

struct MKL: public ComputEnv{
    //this->env_name = "MKL";
    constexpr MKL() {ComputEnv("MKL");};
};
struct MPI: public ComputEnv{
    //this->env_name = "MPI";
    constexpr MPI() {ComputEnv("MPI");};
};
struct CUDA: public ComputEnv{
    //this->env_name = "CUDA";
    constexpr CUDA(){ComputEnv("CUDA"); };
};
struct ROCM: public ComputEnv{
    constexpr ROCM(){ComputEnv("ROCM");};
};

}


