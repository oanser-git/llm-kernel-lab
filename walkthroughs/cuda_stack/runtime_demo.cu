#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)

void check_cuda(cudaError_t result, const char *call, const char *file, int line)
{
    if (result == cudaSuccess)
    {
        return;
    }

    std::cerr << file << ':' << line << ": " << call << " failed: "
              << cudaGetErrorName(result) << " (" << cudaGetErrorString(result) << ")\n";
    std::exit(EXIT_FAILURE);
}

__global__ void write_answer(int *output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        output[0] = 42;
    }
}

int main()
{
    int *device_output = nullptr;
    CHECK_CUDA(cudaMalloc(&device_output, sizeof(int)));

    write_answer<<<1, 1>>>(device_output);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    int host_output = 0;
    CHECK_CUDA(cudaMemcpy(&host_output, device_output, sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(device_output));

    std::cout << "GPU wrote: " << host_output << '\n';
    return host_output == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
