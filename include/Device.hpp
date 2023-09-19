// Device information
// Read https://www.cppstories.com/2018/08/init-string-member/
#pragma once
#include <string>
namespace TH{
class Device{
protected:
    std::string device_info;
public:
    Device(){};
    Device(std::string info): device_info(std::move(info)){};

    const std::string get_device_info(){ return device_info; };
};

class Serial: public Device{
public:
    Serial() : Device("Serial"){};
};
class CPU: public Device{
public:
    CPU() : Device("CPU"){};
};
class CUDA: public Device{
public:
    CUDA() : Device("CUDA"){};
};
class ROCm: public Device{
public:
    ROCm() : Device("ROCm"){};
};

}


