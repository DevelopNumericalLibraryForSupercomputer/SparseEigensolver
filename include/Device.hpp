// Device information
// Currently, only CPU case is implemented
#pragma once
#include <string>
namespace TensorHetero{
class Device{
public:
    const std::string device_info;

    Device(){};
};
}