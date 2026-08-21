// Lab 01: Kernel Launch And Indexing - OpenCode reviewed version

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)

// CUDA functions return error codes, so every CUDA operation is checked.
void check_cuda(cudaError_t result, const char *call, const char *file, int line)
{
    if (result != cudaSuccess)
    {
        std::cerr << file << ':' << line << ": " << call << " failed: "
                  << cudaGetErrorName(result) << " (" << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void store_thread_information(int *global_indices,
                                         int *block_indices,
                                         int *thread_indices,
                                         int *warp_indices,
                                         int *lane_indices,
                                         int problem_size)
{
    // This is the unique index of this thread across the complete grid.
    int global_index = blockIdx.x * blockDim.x + threadIdx.x;

    // The final block may contain extra threads. They must not access memory.
    if (global_index >= problem_size)
    {
        return;
    }

    // Every array uses global_index as WHERE to write because it is unique.
    // The right side is WHAT information this array records about the thread.
    global_indices[global_index] = global_index;
    block_indices[global_index] = blockIdx.x;
    thread_indices[global_index] = threadIdx.x;
    warp_indices[global_index] = threadIdx.x / warpSize;
    lane_indices[global_index] = threadIdx.x % warpSize;
}

int main(int argc, char **argv)
{
    // Defaults: ./lab1 uses 1000 elements and 256 threads per block.
    int problem_size = 1000;
    int threads_per_block = 256;

    // Optional usage: ./lab1 <problem_size> <threads_per_block>
    if (argc == 3)
    {
        try
        {
            problem_size = std::stoi(argv[1]);
            threads_per_block = std::stoi(argv[2]);
        }
        catch (...)
        {
            std::cerr << "Problem size and block size must be integers.\n";
            return EXIT_FAILURE;
        }
    }
    else if (argc != 1)
    {
        std::cerr << "Usage: " << argv[0] << " [problem_size threads_per_block]\n";
        return EXIT_FAILURE;
    }

    if (problem_size < 0 || threads_per_block <= 0)
    {
        std::cerr << "Problem size must be nonnegative and block size must be positive.\n";
        return EXIT_FAILURE;
    }

    cudaDeviceProp device_properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&device_properties, 0));
    if (threads_per_block > device_properties.maxThreadsPerBlock)
    {
        std::cerr << "Block size exceeds the device limit of "
                  << device_properties.maxThreadsPerBlock << ".\n";
        return EXIT_FAILURE;
    }

    // CUDA cannot launch zero blocks, so an empty problem finishes on the CPU.
    if (problem_size == 0)
    {
        std::cout << "Problem size is zero; no GPU work is required.\n";
        return EXIT_SUCCESS;
    }

    // Ceiling division creates enough whole blocks to cover the problem.
    int blocks_per_grid =
        (problem_size + threads_per_block - 1) / threads_per_block;

    // Indexes are integers, so each output array stores int values.
    std::size_t bytes = static_cast<std::size_t>(problem_size) * sizeof(int);

    int *d_global = nullptr;
    int *d_block = nullptr;
    int *d_thread = nullptr;
    int *d_warp = nullptr;
    int *d_lane = nullptr;

    // cudaMalloc expects bytes, not a number of elements.
    CHECK_CUDA(cudaMalloc(&d_global, bytes));
    CHECK_CUDA(cudaMalloc(&d_block, bytes));
    CHECK_CUDA(cudaMalloc(&d_thread, bytes));
    CHECK_CUDA(cudaMalloc(&d_warp, bytes));
    CHECK_CUDA(cudaMalloc(&d_lane, bytes));

    store_thread_information<<<blocks_per_grid, threads_per_block>>>(
        d_global, d_block, d_thread, d_warp, d_lane, problem_size);

    // Check whether the launch was valid and whether execution succeeded.
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_global(problem_size);
    std::vector<int> h_block(problem_size);
    std::vector<int> h_thread(problem_size);
    std::vector<int> h_warp(problem_size);
    std::vector<int> h_lane(problem_size);

    // These are output arrays, so only Device-to-Host copies are required.
    CHECK_CUDA(cudaMemcpy(h_global.data(), d_global, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_block.data(), d_block, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_thread.data(), d_thread, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_warp.data(), d_warp, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_lane.data(), d_lane, bytes, cudaMemcpyDeviceToHost));

    // Validate every value using an independent CPU calculation.
    for (int global = 0; global < problem_size; ++global)
    {
        int expected_block = global / threads_per_block;
        int expected_thread = global % threads_per_block;
        int expected_warp = expected_thread / device_properties.warpSize;
        int expected_lane = expected_thread % device_properties.warpSize;

        if (h_global[global] != global ||
            h_block[global] != expected_block ||
            h_thread[global] != expected_thread ||
            h_warp[global] != expected_warp ||
            h_lane[global] != expected_lane)
        {
            std::cerr << "Validation failed for global thread " << global << ".\n";
            return EXIT_FAILURE;
        }
    }

    // Print a small table so the hierarchy is visible without huge output.
    // Up to 70 rows reveal the lane wrap at 32 and, with block size 50 or 64,
    // the point where thread and warp indexes restart in the next block.
    int rows_to_print = std::min(problem_size, 70);
    std::cout << "global  block  thread  warp  lane\n";
    for (int global = 0; global < rows_to_print; ++global)
    {
        std::cout << h_global[global] << "\t"
                  << h_block[global] << "\t"
                  << h_thread[global] << "\t"
                  << h_warp[global] << "\t"
                  << h_lane[global] << '\n';
    }

    CHECK_CUDA(cudaFree(d_global));
    CHECK_CUDA(cudaFree(d_block));
    CHECK_CUDA(cudaFree(d_thread));
    CHECK_CUDA(cudaFree(d_warp));
    CHECK_CUDA(cudaFree(d_lane));

    std::cout << "Validated all " << problem_size << " threads using "
              << blocks_per_grid << " blocks x " << threads_per_block << " threads.\n";
    return EXIT_SUCCESS;
}

// README QUESTION 1:
// Extra threads are safe because the bounds check returns before they access an
// array when global_index >= problem_size.

// README QUESTION 2:
// A block is a programmer-defined group that can share memory and synchronize.
// A warp is a 32-lane hardware scheduling group inside one block.

// README QUESTION 3:
// In a grid-stride loop, each thread advances by gridDim.x * blockDim.x. A fixed
// number of launched threads can therefore process a much larger problem.

// README QUESTION 4:
// Kernel launches are asynchronous. Without synchronization, host wall-clock
// timing can measure only launch submission while the GPU is still executing.
