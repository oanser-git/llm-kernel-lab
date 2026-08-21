#include <cuda.h>

#include <cstdlib>
#include <iostream>

#define CHECK_DRIVER(call) check_driver((call), #call, __FILE__, __LINE__)

void check_driver(CUresult result, const char *call, const char *file, int line)
{
    if (result == CUDA_SUCCESS)
    {
        return;
    }

    const char *name = "unknown";
    const char *description = "unknown";
    cuGetErrorName(result, &name);
    cuGetErrorString(result, &description);
    std::cerr << file << ':' << line << ": " << call << " failed: " << name << " ("
              << description << ")\n";
    std::exit(EXIT_FAILURE);
}

int main()
{
    CHECK_DRIVER(cuInit(0));

    int driver_version = 0;
    int device_count = 0;
    CHECK_DRIVER(cuDriverGetVersion(&driver_version));
    CHECK_DRIVER(cuDeviceGetCount(&device_count));

    CUdevice device{};
    char device_name[256]{};
    CHECK_DRIVER(cuDeviceGet(&device, 0));
    CHECK_DRIVER(cuDeviceGetName(device_name, sizeof(device_name), device));

    std::cout << "Driver API version: " << driver_version / 1000 << '.'
              << (driver_version % 1000) / 10 << '\n';
    std::cout << "Driver API devices: " << device_count << '\n';
    std::cout << "Driver API device 0: " << device_name << '\n';
    return EXIT_SUCCESS;
}
