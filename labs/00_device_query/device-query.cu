// Lab 00: Device Query - student version before OpenCode completion

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char *function, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA error at " << file << ':' << line << " while calling " << function
                  << ": " << cudaGetErrorName(result) << " (" << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

int main(void)
{
    int deviceId = 0;
    cudaDeviceProp deviceProp;

    CHECK_CUDA(cudaGetDeviceProperties(&deviceProp, deviceId));

    std::cout << "Device Name: " << deviceProp.name << "\n";
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
    std::cout << "Multiprocessors (SMs): " << deviceProp.multiProcessorCount << "\n";
    // totalGlobalMem is measured in bytes:
    // 1 KiB = 1024 bytes
    // 1 MiB = 1024 KiB
    // 1 MiB = 1024 x 1024 bytes
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MiB\n";
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << "\n";

    return 0;
}
