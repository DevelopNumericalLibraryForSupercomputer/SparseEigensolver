// Comput Environment information
#pragma once
#include <string>
namespace SE{
struct ComputEnv{
    public:
    const char* env_name;
    constexpr ComputEnv(const char* _env_name="BASE"): env_name(_env_name){};
};

struct MKL: public ComputEnv{
    constexpr MKL() {ComputEnv("MKL");};
};
struct MPI: public ComputEnv{
    constexpr MPI() {ComputEnv("MPI");};
};
struct CUDA: public ComputEnv{
    constexpr CUDA(){ComputEnv("CUDA"); };
};
struct ROCM: public ComputEnv{
    constexpr ROCM(){ComputEnv("ROCM");};
};

}


