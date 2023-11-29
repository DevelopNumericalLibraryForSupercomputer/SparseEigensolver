// Comput Environment information
#pragma once
#include <string>
namespace SE{
struct ComputEnv{
    public:
    const char* env_name;
    constexpr ComputEnv(const char* _env_name="BASE"): env_name(_env_name){};
};

struct SEMkl: public ComputEnv{
    constexpr SEMkl() {ComputEnv("MKL");};
};
struct SEMpi: public ComputEnv{
    constexpr SEMpi() {ComputEnv("MPI");};
};
struct SECuda: public ComputEnv{
    constexpr SECuda(){ComputEnv("CUDA"); };
};
struct SENccl: public ComputEnv{
    constexpr SENccl(){ComputEnv("NCCL");};
};

}


