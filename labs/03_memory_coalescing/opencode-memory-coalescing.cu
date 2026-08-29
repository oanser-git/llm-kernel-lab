// Lab 03: Memory Coalescing - OpenCode reviewed checkpoint 1
// This establishes the contiguous transform used as the control experiment.

#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void contiguous_transform(const float *input, float *output, int problem_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < problem_size)
    {
        output[index] = input[index] * 2.0F + 1.0F;
    }
}

void initialize_input(std::vector<float> &input)
{
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(index) * 0.5F;
    }
}

void calculate_reference(const std::vector<float> &input, std::vector<float> &reference)
{
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        reference[index] = input[index] * 2.0F + 1.0F;
    }
}

bool validate_output(const std::vector<float> &output, const std::vector<float> &reference)
{
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        if (output[index] != reference[index])
        {
            std::cerr << "Validation failed at index " << index << ": expected " << reference[index]
                      << ", received " << output[index] << '\n';
            return false;
        }
    }

    return true;
}

int main(int argc, char **argv)
{
    // Usage: ./opencode-memory-coalescing [problem_size] [threads_per_block]
    int problem_size = argc >= 2 ? std::atoi(argv[1]) : 1000;
    int threads_per_block = argc >= 3 ? std::atoi(argv[2]) : 256;

    if (argc > 3 || problem_size < 0 || threads_per_block <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [nonnegative_problem_size] [positive_threads_per_block]\n";
        return EXIT_FAILURE;
    }

    if (problem_size == 0)
    {
        PRINT("Problem size is zero; no GPU work is required.");
        return EXIT_SUCCESS;
    }

    std::vector<float> h_input(problem_size);
    std::vector<float> h_output(problem_size, -1.0F);
    std::vector<float> h_reference(problem_size);

    initialize_input(h_input);
    calculate_reference(h_input, h_reference);

    std::size_t bytes = static_cast<std::size_t>(problem_size) * sizeof(float);

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, bytes));
    CHECK_CUDA(cudaMalloc(&d_output, bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice));

    int blocks_per_grid = (problem_size - 1) / threads_per_block + 1;
    contiguous_transform<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, problem_size);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    if (!validate_output(h_output, h_reference))
    {
        return EXIT_FAILURE;
    }

    int print_limit = std::min(problem_size, 10);
    for (int index = 0; index < print_limit; ++index)
    {
        PRINT("output[{}] = {}", index, h_output[index]);
    }

    PRINT("Validated all {} contiguous outputs using {} blocks x {} threads.", problem_size,
          blocks_per_grid, threads_per_block);
    return EXIT_SUCCESS;
}
