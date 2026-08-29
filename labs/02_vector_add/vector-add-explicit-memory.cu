// Lab 02: Vector Addition with explicit CUDA device memory

#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstdio>

__global__ void vector_add(const float *a, const float *b, float *output, int problem_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < problem_size)
    {
        output[idx] = a[idx] + b[idx];
    }
}

__global__ void vector_add_stride_step(const float *a, const float *b, float *output,
                                       int problem_size)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < problem_size; i += stride)
    {
        output[i] = a[i] + b[i];
    }
}

void initialize_inputs(std::vector<float> &h_a, std::vector<float> &h_b,
                       std::vector<float> &h_reference, int problem_size)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < problem_size; ++i)
    {
        h_a[i] = dist(gen);
        h_b[i] = dist(gen);
        h_reference[i] = h_a[i] + h_b[i];
    }
}

float benchmark_scalar_kernel(const float *d_a, const float *d_b, float *d_output, int problem_size,
                              int threads_per_block, int warmup_iterations, int repetitions,
                              cudaEvent_t start, cudaEvent_t stop)
{
    int blocks_per_grid = (problem_size + threads_per_block - 1) / threads_per_block;

    dim3 block(threads_per_block, 1, 1);
    dim3 grid(blocks_per_grid, 1, 1);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        vector_add<<<grid, block>>>(d_a, d_b, d_output, problem_size);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        vector_add<<<grid, block>>>(d_a, d_b, d_output, problem_size);
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));

    return total_milliseconds / static_cast<float>(repetitions);
}

float benchmark_grid_stride_kernel(const float *d_a, const float *d_b, float *d_output,
                                   int problem_size, int threads_per_block, int blocks_per_grid,
                                   int warmup_iterations, int repetitions, cudaEvent_t start,
                                   cudaEvent_t stop)
{
    dim3 block(threads_per_block, 1, 1);
    dim3 grid(blocks_per_grid, 1, 1);

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        vector_add_stride_step<<<grid, block>>>(d_a, d_b, d_output, problem_size);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        vector_add_stride_step<<<grid, block>>>(d_a, d_b, d_output, problem_size);
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));

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
    CHECK_CUDA(cudaEventRecord(start, 0));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        CHECK_CUDA(cudaMemcpyAsync(d_destination, d_source, bytes, cudaMemcpyDeviceToDevice));
    }

    CHECK_CUDA(cudaEventRecord(stop, 0));
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

bool validate_output(const std::vector<float> &h_output, const std::vector<float> &h_reference,
                     int problem_size)
{
    for (int index = 0; index < problem_size; ++index)
    {
        if (h_output[index] != h_reference[index])
        {
            std::cerr << "Validation failed at index " << index << ": expected "
                      << h_reference[index] << ", received " << h_output[index] << '\n';
            return false;
        }
    }

    return true;
}

int main()
{
    int problem_size = 10000000;
    int warmup_iterations = 10;
    int repetitions = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int device_id = 0;
    cudaDeviceProp device_properties{};
    CHECK_CUDA(cudaGetDevice(&device_id));
    CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_id));

    // Launch a fixed grid targeting four blocks per SM.
    // Grid-stride loops let these blocks process the entire vector.
    int grid_stride_blocks = device_properties.multiProcessorCount * 4;

    // host part
    std::vector<float> h_a(problem_size);
    std::vector<float> h_b(problem_size);
    std::vector<float> h_reference(problem_size);
    std::vector<float> h_scalar_output(problem_size, -1.0F);
    std::vector<float> h_grid_stride_output(problem_size, -1.0F);
    std::vector<float> h_copy_output(problem_size, -1.0F);

    initialize_inputs(h_a, h_b, h_reference, problem_size);

    std::size_t bytes = static_cast<std::size_t>(problem_size) * sizeof(float);

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

    double useful_bytes_processed = 3.0 * static_cast<double>(problem_size) * sizeof(float);
    double useful_bytes_copied = 2.0 * static_cast<double>(problem_size) * sizeof(float);

    bool validation_passed = true;
    const int block_sizes[] = {128, 256};
    for (int threads_per_block : block_sizes)
    {
        float scalar_milliseconds =
            benchmark_scalar_kernel(d_a, d_b, d_scalar_output, problem_size, threads_per_block,
                                    warmup_iterations, repetitions, start, stop);

        float grid_stride_milliseconds = benchmark_grid_stride_kernel(
            d_a, d_b, d_grid_stride_output, problem_size, threads_per_block, grid_stride_blocks,
            warmup_iterations, repetitions, start, stop);

        double scalar_bandwidth =
            calculate_effective_bandwidth(useful_bytes_processed, scalar_milliseconds);
        double grid_stride_bandwidth =
            calculate_effective_bandwidth(useful_bytes_processed, grid_stride_milliseconds);

        PRINT("block size: {}", threads_per_block);
        PRINT("scalar: {} ms, {} GB/s", scalar_milliseconds, scalar_bandwidth);
        PRINT("grid-stride: {} ms, {} GB/s", grid_stride_milliseconds, grid_stride_bandwidth);

        CHECK_CUDA(
            cudaMemcpy(h_scalar_output.data(), d_scalar_output, bytes, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_grid_stride_output.data(), d_grid_stride_output, bytes,
                              cudaMemcpyDeviceToHost));

        bool scalar_valid = validate_output(h_scalar_output, h_reference, problem_size);
        bool grid_stride_valid = validate_output(h_grid_stride_output, h_reference, problem_size);
        validation_passed = validation_passed && scalar_valid && grid_stride_valid;
    }

    float copy_milliseconds = benchmark_device_copy(d_a, d_copy_output, bytes, warmup_iterations,
                                                    repetitions, start, stop);
    double copy_bandwidth = calculate_effective_bandwidth(useful_bytes_copied, copy_milliseconds);

    PRINT("D2D copy: {} ms, {} GB/s", copy_milliseconds, copy_bandwidth);

    CHECK_CUDA(cudaMemcpy(h_copy_output.data(), d_copy_output, bytes, cudaMemcpyDeviceToHost));

    bool copy_valid = validate_output(h_copy_output, h_a, problem_size);
    validation_passed = validation_passed && copy_valid;

    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_scalar_output));
    CHECK_CUDA(cudaFree(d_grid_stride_output));
    CHECK_CUDA(cudaFree(d_copy_output));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return validation_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
