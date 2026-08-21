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

    CUdevice device{};
    CHECK_DRIVER(cuDeviceGet(&device, 0));

    CUcontext context{};
    CHECK_DRIVER(cuCtxCreate(&context, nullptr, 0, device));

    CUmodule module{};
    CHECK_DRIVER(cuModuleLoad(&module, "driver_kernel.cubin"));

    CUfunction kernel{};
    CHECK_DRIVER(cuModuleGetFunction(&kernel, module, "write_answer"));

    CUdeviceptr device_output{};
    CHECK_DRIVER(cuMemAlloc(&device_output, sizeof(int)));

    void *kernel_arguments[] = {&device_output};
    CHECK_DRIVER(cuLaunchKernel(kernel,
                                1, 1, 1,
                                1, 1, 1,
                                0,
                                nullptr,
                                kernel_arguments,
                                nullptr));
    CHECK_DRIVER(cuCtxSynchronize());

    int host_output = 0;
    CHECK_DRIVER(cuMemcpyDtoH(&host_output, device_output, sizeof(int)));

    CHECK_DRIVER(cuMemFree(device_output));
    CHECK_DRIVER(cuModuleUnload(module));
    CHECK_DRIVER(cuCtxDestroy(context));

    std::cout << "GPU wrote through Driver API: " << host_output << '\n';
    return host_output == 42 ? EXIT_SUCCESS : EXIT_FAILURE;
}
