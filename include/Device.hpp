// Device information
#pragma once
#include <string>
namespace TH{
class Device{
private:
    const std::string device_info;
public:
    Device(){};
    Device(const std::string info):device_info(info){};
    
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


