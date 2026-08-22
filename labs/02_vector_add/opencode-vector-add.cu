// Lab 02: Vector Addition - OpenCode reviewed required checkpoints
// This version uses the portable H2D -> kernel -> D2H memory flow. It includes
// scalar and grid-stride kernels, reliable timing, useful bandwidth, and D2D.

#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

__global__ void vector_add(const float *a, const float *b, float *output, int problem_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < problem_size)
    {
        output[index] = a[index] + b[index];
    }
}

__global__ void vector_add_grid_stride(const float *a, const float *b, float *output,
                                       int problem_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (; index < problem_size; index += stride)
    {
        output[index] = a[index] + b[index];
    }
}

void initialize_inputs(std::vector<float> &h_a, std::vector<float> &h_b,
                       std::vector<float> &h_reference, int problem_size)
{
    for (int index = 0; index < problem_size; ++index)
    {
        h_a[index] = static_cast<float>(index) * 0.5F;
        h_b[index] = static_cast<float>(index) * 0.25F + 1.0F;
        h_reference[index] = h_a[index] + h_b[index];
    }
}

float benchmark_scalar_kernel(const float *d_a, const float *d_b, float *d_output, int problem_size,
                              int threads_per_block, int warmup_iterations, int repetitions,
                              cudaEvent_t start, cudaEvent_t stop)
{
    int blocks_per_grid = (problem_size - 1) / threads_per_block + 1;

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_output, problem_size);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        vector_add<<<blocks_per_grid, threads_per_block>>>(d_a, d_b, d_output, problem_size);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, start, stop));

    return total_milliseconds / static_cast<float>(repetitions);
}

float benchmark_grid_stride_kernel(const float *d_a, const float *d_b, float *d_output,
                                   int problem_size, int threads_per_block, int grid_stride_blocks,
                                   int warmup_iterations, int repetitions, cudaEvent_t start,
                                   cudaEvent_t stop)
{
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        vector_add_grid_stride<<<grid_stride_blocks, threads_per_block>>>(d_a, d_b, d_output,
                                                                          problem_size);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        vector_add_grid_stride<<<grid_stride_blocks, threads_per_block>>>(d_a, d_b, d_output,
                                                                          problem_size);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, start, stop));

    return total_milliseconds / static_cast<float>(repetitions);
}

float benchmark_device_copy(const float *d_source, float *d_destination, std::size_t bytes,
                            int warmup_iterations, int repetitions, cudaEvent_t start,
                            cudaEvent_t stop)
{
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        CHECK_CUDA(cudaMemcpyAsync(d_destination, d_source, bytes, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(start));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        CHECK_CUDA(cudaMemcpyAsync(d_destination, d_source, bytes, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CHECK_CUDA(cudaEventElapsedTime(&total_milliseconds, start, stop));

    return total_milliseconds / static_cast<float>(repetitions);
}

double calculate_effective_bandwidth(double useful_bytes, float average_milliseconds)
{
    double elapsed_seconds = static_cast<double>(average_milliseconds) / 1000.0;

    return useful_bytes / elapsed_seconds / 1.0e9;
}

bool validate_output(const std::string &label, const std::vector<float> &output,
                     const std::vector<float> &reference)
{
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        if (output[index] != reference[index])
        {
            std::cerr << label << " validation failed at index " << index << ": expected "
                      << reference[index] << ", received " << output[index] << '\n';
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    // Usage: ./opencode-vector-add [problem_size] [grid_stride_blocks] [warmups] [repetitions]
    int problem_size = argc >= 2 ? std::atoi(argv[1]) : 10000000;
    int requested_grid_stride_blocks = argc >= 3 ? std::atoi(argv[2]) : 0;
    int warmup_iterations = argc >= 4 ? std::atoi(argv[3]) : 10;
    int repetitions = argc >= 5 ? std::atoi(argv[4]) : 100;

    if (argc > 5 || problem_size < 0 || (argc >= 3 && requested_grid_stride_blocks <= 0) ||
        warmup_iterations <= 0 || repetitions <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [nonnegative_problem_size] [positive_grid_stride_blocks]"
                  << " [positive_warmups] [positive_repetitions]\n";
        return EXIT_FAILURE;
    }

    if (problem_size == 0)
    {
        std::cout << "Problem size is zero; no GPU work is required.\n";
        return EXIT_SUCCESS;
    }

    int device_id = 0;
    cudaDeviceProp device_properties{};
    CHECK_CUDA(cudaGetDevice(&device_id));
    CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_id));

    int grid_stride_blocks = requested_grid_stride_blocks > 0
                                 ? requested_grid_stride_blocks
                                 : device_properties.multiProcessorCount * 4;

    std::vector<float> h_a(problem_size);
    std::vector<float> h_b(problem_size);
    std::vector<float> h_reference(problem_size);
    std::vector<float> h_scalar_output(problem_size, -1.0F);
    std::vector<float> h_grid_stride_output(problem_size, -1.0F);
    std::vector<float> h_copy_output(problem_size, -1.0F);

    initialize_inputs(h_a, h_b, h_reference, problem_size);

    std::size_t bytes = static_cast<std::size_t>(problem_size) * sizeof(float);
    double vector_add_useful_bytes = 3.0 * static_cast<double>(bytes);
    double copy_useful_bytes = 2.0 * static_cast<double>(bytes);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_scalar_output = nullptr;
    float *d_grid_stride_output = nullptr;
    float *d_copy_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_scalar_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_grid_stride_output, bytes));
    CHECK_CUDA(cudaMalloc(&d_copy_output, bytes));

    CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const int block_sizes[] = {128, 256};
    for (int threads_per_block : block_sizes)
    {
        float scalar_milliseconds =
            benchmark_scalar_kernel(d_a, d_b, d_scalar_output, problem_size, threads_per_block,
                                    warmup_iterations, repetitions, start, stop);
        float grid_stride_milliseconds = benchmark_grid_stride_kernel(
            d_a, d_b, d_grid_stride_output, problem_size, threads_per_block, grid_stride_blocks,
            warmup_iterations, repetitions, start, stop);

        CHECK_CUDA(
            cudaMemcpy(h_scalar_output.data(), d_scalar_output, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_grid_stride_output.data(), d_grid_stride_output, bytes,
                              cudaMemcpyDeviceToHost));

        if (!validate_output("Scalar", h_scalar_output, h_reference) ||
            !validate_output("Grid-stride", h_grid_stride_output, h_reference))
        {
            return EXIT_FAILURE;
        }

        double scalar_bandwidth =
            calculate_effective_bandwidth(vector_add_useful_bytes, scalar_milliseconds);
        double grid_stride_bandwidth =
            calculate_effective_bandwidth(vector_add_useful_bytes, grid_stride_milliseconds);

        std::cout << "Block size " << threads_per_block << '\n'
                  << "  Scalar: " << scalar_milliseconds << " ms, " << scalar_bandwidth << " GB/s\n"
                  << "  Grid-stride (" << grid_stride_blocks
                  << " blocks): " << grid_stride_milliseconds << " ms, " << grid_stride_bandwidth
                  << " GB/s\n";
    }

    float copy_milliseconds = benchmark_device_copy(d_a, d_copy_output, bytes, warmup_iterations,
                                                    repetitions, start, stop);
    CHECK_CUDA(cudaMemcpy(h_copy_output.data(), d_copy_output, bytes, cudaMemcpyDeviceToHost));

    if (!validate_output("D2D copy", h_copy_output, h_a))
    {
        return EXIT_FAILURE;
    }

    double copy_bandwidth = calculate_effective_bandwidth(copy_useful_bytes, copy_milliseconds);
    std::cout << "D2D copy: " << copy_milliseconds << " ms, " << copy_bandwidth << " GB/s\n";

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_scalar_output));
    CHECK_CUDA(cudaFree(d_grid_stride_output));
    CHECK_CUDA(cudaFree(d_copy_output));

    std::cout << "Validated all " << problem_size
              << " elements for both kernels and the D2D baseline.\n";
    return EXIT_SUCCESS;
}
