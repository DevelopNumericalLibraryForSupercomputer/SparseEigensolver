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

class CPU: public Device{
public:
    CPU() : Device("CPU"){};
};

class GPU: public Device{
public:
    GPU() : Device("GPU"){};
};

}


