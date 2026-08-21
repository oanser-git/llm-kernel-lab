// Lab 00: Device Query - OpenCode reviewed version

#include <cuda_runtime.h>

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <system_error>

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

bool parse_device_id(std::string_view text, int &device_id)
{
    const char *begin = text.data();
    const char *end = begin + text.size();
    const auto result = std::from_chars(begin, end, device_id);
    return result.ec == std::errc{} && result.ptr == end;
}

const char *yes_no(int value)
{
    return value != 0 ? "yes" : "no";
}

void print_cuda_version(const char *label, int version)
{
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;
    std::cout << label << major << '.' << minor << '\n';
}

int main(int argc, char **argv)
{
    if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [device_id]\n";
        return EXIT_FAILURE;
    }

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices were found.\n";
        return EXIT_FAILURE;
    }

    int device_id = 0;
    if (argc == 2 && !parse_device_id(argv[1], device_id))
    {
        std::cerr << "Invalid device ID: " << argv[1] << "\n";
        return EXIT_FAILURE;
    }
    if (device_id < 0 || device_id >= device_count)
    {
        std::cerr << "Device ID " << device_id << " is out of range. Available IDs: 0-"
                  << device_count - 1 << "\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device_id));

    cudaDeviceProp device_prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&device_prop, device_id));

    int driver_version = 0;
    int runtime_version = 0;
    int core_clock_khz = 0;
    int memory_clock_khz = 0;
    CHECK_CUDA(cudaDriverGetVersion(&driver_version));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime_version));
    CHECK_CUDA(cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device_id));
    CHECK_CUDA(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device_id));

    constexpr double bytes_per_kib = 1024.0;
    constexpr double bytes_per_mib = bytes_per_kib * 1024.0;
    constexpr double bytes_per_gib = bytes_per_mib * 1024.0;

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "CUDA devices: " << device_count << '\n';
    std::cout << "Selected device: " << device_id << '\n';
    print_cuda_version("CUDA driver API version: ", driver_version);
    print_cuda_version("CUDA runtime version: ", runtime_version);

    std::cout << "\nIdentity and execution\n";
    std::cout << "  Name: " << device_prop.name << '\n';
    std::cout << "  Compute capability: " << device_prop.major << '.' << device_prop.minor << '\n';
    std::cout << "  Multiprocessors (SMs): " << device_prop.multiProcessorCount << '\n';
    std::cout << "  Warp size: " << device_prop.warpSize << '\n';
    std::cout << "  Maximum threads per block: " << device_prop.maxThreadsPerBlock << '\n';
    std::cout << "  Maximum block dimensions: " << device_prop.maxThreadsDim[0] << " x "
              << device_prop.maxThreadsDim[1] << " x " << device_prop.maxThreadsDim[2] << '\n';
    std::cout << "  Maximum grid dimensions: " << device_prop.maxGridSize[0] << " x "
              << device_prop.maxGridSize[1] << " x " << device_prop.maxGridSize[2] << '\n';
    std::cout << "  Maximum threads per SM: " << device_prop.maxThreadsPerMultiProcessor << '\n';
    std::cout << "  Maximum blocks per SM: " << device_prop.maxBlocksPerMultiProcessor << '\n';
    std::cout << "  Concurrent kernel execution: " << yes_no(device_prop.concurrentKernels) << '\n';
    std::cout << "  Asynchronous copy engines: " << device_prop.asyncEngineCount << '\n';

    std::cout << "\nCompute resources\n";
    if (core_clock_khz > 0)
    {
        std::cout << "  Maximum core clock: " << core_clock_khz / 1000.0 << " MHz\n";
    }
    else
    {
        std::cout << "  Maximum core clock: unavailable\n";
    }
    std::cout << "  Registers per block: " << device_prop.regsPerBlock << '\n';
    std::cout << "  Registers per SM: " << device_prop.regsPerMultiprocessor << '\n';
    std::cout << "  Shared memory per block: "
              << device_prop.sharedMemPerBlock / bytes_per_kib << " KiB\n";
    std::cout << "  Opt-in shared memory per block: "
              << device_prop.sharedMemPerBlockOptin / bytes_per_kib << " KiB\n";
    std::cout << "  Shared memory per SM: "
              << device_prop.sharedMemPerMultiprocessor / bytes_per_kib << " KiB\n";

    std::cout << "\nMemory\n";
    std::cout << "  Global memory: " << device_prop.totalGlobalMem / bytes_per_mib << " MiB ("
              << device_prop.totalGlobalMem / bytes_per_gib << " GiB)\n";
    std::cout << "  L2 cache: " << device_prop.l2CacheSize / bytes_per_mib << " MiB\n";
    if (memory_clock_khz > 0)
    {
        std::cout << "  Maximum memory clock: " << memory_clock_khz / 1000.0 << " MHz\n";
    }
    else
    {
        std::cout << "  Maximum memory clock: unavailable\n";
    }
    if (device_prop.memoryBusWidth > 0)
    {
        std::cout << "  Memory bus width: " << device_prop.memoryBusWidth << " bits\n";
    }
    else
    {
        std::cout << "  Memory bus width: unavailable\n";
    }
    std::cout << "  Memory bandwidth: not derived; transfer-rate interpretation is platform-specific\n";

    std::cout << "\nAddressing and managed memory\n";
    std::cout << "  Unified virtual addressing: " << yes_no(device_prop.unifiedAddressing) << '\n';
    std::cout << "  Managed memory: " << yes_no(device_prop.managedMemory) << '\n';
    std::cout << "  Concurrent managed access: " << yes_no(device_prop.concurrentManagedAccess) << '\n';
    std::cout << "  Pageable memory access: " << yes_no(device_prop.pageableMemoryAccess) << '\n';
    std::cout << "  Can map host memory: " << yes_no(device_prop.canMapHostMemory) << '\n';
    std::cout << "  Memory pools supported: " << yes_no(device_prop.memoryPoolsSupported) << '\n';

    return EXIT_SUCCESS;
}
