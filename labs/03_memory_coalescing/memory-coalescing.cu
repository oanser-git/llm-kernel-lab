// Lab 03: Memory Coalescing
// Read README.md, then implement the lab from scratch in this file.

#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

__global__ void scalar_transform_kernel(const float *__restrict__ input, float *__restrict__ output,
                                        int problem_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < problem_size)
    {
        output[index] = input[index] * 2.0F + 1.0F;
    }
}

__global__ void stride_transform_kernel(const float *__restrict__ input, float *__restrict__ output,
                                        int problem_size, int memory_stride)
{
    int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index < problem_size)
    {
        int input_index = output_index * memory_stride;
        output[output_index] = input[input_index] * 2.0F + 1.0F;
    }
}

__global__ void offset_scalar_transform_kernel(const float *__restrict__ input,
                                               float *__restrict__ output, int problem_size,
                                               int offset)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < problem_size)
    {
        output[index] = input[index + offset] * 2.0F + 1.0F;
    }
}

__global__ void gather_kernel(const float *__restrict__ input, float *__restrict__ output,
                              const int *__restrict__ indices, int problem_size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    for (int i = index; i < problem_size; i += grid_stride)
    {
        int fetch_index = indices[i];
        output[i] = input[fetch_index] * 2.0F + 1.0F;
    }
}

void initialize_input(std::vector<float> &input)
{
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(index) * 0.5F;
    }
}

void calculate_reference(const std::vector<float> &input, std::vector<float> &reference, int offset)
{
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        reference[index] = input[index + offset] * 2.0F + 1.0F;
    }
}

void calculate_stride_reference(const std::vector<float> &input, std::vector<float> &reference,
                                int memory_stride)
{
    for (std::size_t output_index = 0; output_index < reference.size(); ++output_index)
    {
        std::size_t input_index = output_index * memory_stride;
        reference[output_index] = input[input_index] * 2.0F + 1.0F;
    }
}

void initialize_gather_indices(std::vector<int> &indices, int input_size, int pattern)
{
    for (std::size_t index = 0; index < indices.size(); ++index)
    {
        if (pattern == 0)
        {
            indices[index] = static_cast<int>(index);
        }
        else if (pattern == 1)
        {
            bool is_even = index % 2 == 0;
            bool has_next = index + 1 < indices.size();
            indices[index] = is_even && has_next ? static_cast<int>(index + 1)
                                                 : static_cast<int>(is_even ? index : index - 1);
        }
        else
        {
            indices[index] = static_cast<int>((index * 131U) % input_size);
        }
    }
}

void calculate_gather_reference(const std::vector<float> &input, const std::vector<int> &indices,
                                std::vector<float> &reference)
{
    for (std::size_t index = 0; index < reference.size(); ++index)
    {
        reference[index] = input[indices[index]] * 2.0F + 1.0F;
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

template <typename KernelFunc, typename... Args>
float benchmark_kernel(KernelFunc kernel, int threads_per_block, int blocks_per_grid,
                       int warmup_iterations, int repetitions, cudaEvent_t start, cudaEvent_t stop,
                       Args... args)
{

    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        kernel<<<blocks_per_grid, threads_per_block>>>(args...);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, 0));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        kernel<<<blocks_per_grid, threads_per_block>>>(args...);
    }

    CUDA_CHECK(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));

    return total_milliseconds / static_cast<float>(repetitions);
}

double calculate_effective_bandwidth(double useful_bytes, float average_milliseconds)
{
    double elapsed_seconds = static_cast<double>(average_milliseconds) / 1000.0;

    return useful_bytes / elapsed_seconds / 1.0e9;
}

int main(int argc, char **argv)
{
    // Gather patterns: 0 = contiguous, 1 = neighboring pairs, 2 = scattered.
    // Usage: ./memory-coalescing [problem_size] [threads_per_block] [offset] [memory_stride]
    //                            [gather_pattern] [warmups] [repetitions]
    int problem_size = argc >= 2 ? std::atoi(argv[1]) : 10000000;
    int threads_per_block = argc >= 3 ? std::atoi(argv[2]) : 256;
    int offset = argc >= 4 ? std::atoi(argv[3]) : 2;
    int memory_stride = argc >= 5 ? std::atoi(argv[4]) : 2;
    int gather_pattern = argc >= 6 ? std::atoi(argv[5]) : 2;
    int warmup_iterations = argc >= 7 ? std::atoi(argv[6]) : 10;
    int repetitions = argc >= 8 ? std::atoi(argv[7]) : 100;

    if (argc > 8 || problem_size < 0 || threads_per_block <= 0 || offset < 0 ||
        memory_stride <= 0 || gather_pattern < 0 || gather_pattern > 2 || warmup_iterations <= 0 ||
        repetitions <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [nonnegative_problem_size] [positive_threads_per_block]"
                  << " [nonnegative_offset] [positive_memory_stride]"
                  << " [gather_pattern_0_to_2] [positive_warmups]"
                  << " [positive_repetitions]\n";
        return EXIT_FAILURE;
    }

    if (problem_size == 0)
    {
        PRINT("Problem size is zero; no GPU work is required.");
        return EXIT_SUCCESS;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int offset_input_size = problem_size + offset;
    int stride_input_size = (problem_size - 1) * memory_stride + 1;
    int input_size = std::max(offset_input_size, stride_input_size);

    std::vector<float> h_input(input_size);
    std::vector<float> h_output(problem_size, -1.0F);
    std::vector<float> h_reference(problem_size);
    std::vector<int> h_indices(problem_size);

    initialize_input(h_input);

    std::size_t input_bytes = static_cast<std::size_t>(input_size) * sizeof(float);
    std::size_t output_bytes = static_cast<std::size_t>(problem_size) * sizeof(float);
    std::size_t indices_bytes = static_cast<std::size_t>(problem_size) * sizeof(int);

    float *d_input = nullptr;
    float *d_output = nullptr;
    int *d_indices = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_indices, indices_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    int blocks_per_grid = (problem_size - 1) / threads_per_block + 1;
    double transform_useful_bytes = 2.0 * static_cast<double>(problem_size) * sizeof(float);
    double gather_useful_bytes = 3.0 * static_cast<double>(problem_size) * sizeof(float);

    // -----------------------------------------------------------------------
    calculate_reference(h_input, h_reference, 0);

    float scalar_transform_milliseconds = benchmark_kernel(
        scalar_transform_kernel, threads_per_block, blocks_per_grid, warmup_iterations, repetitions,
        start, stop, d_input, d_output, problem_size);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    double scalar_transform_bandwidth =
        calculate_effective_bandwidth(transform_useful_bytes, scalar_transform_milliseconds);

    bool transform_valid = validate_output(h_output, h_reference);
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    calculate_reference(h_input, h_reference, offset);

    float offset_scalar_transform_milliseconds = benchmark_kernel(
        offset_scalar_transform_kernel, threads_per_block, blocks_per_grid, warmup_iterations,
        repetitions, start, stop, d_input, d_output, problem_size, offset);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    double offset_scalar_transform_bandwidth =
        calculate_effective_bandwidth(transform_useful_bytes, offset_scalar_transform_milliseconds);

    bool offset_valid = validate_output(h_output, h_reference);
    // ------------------------------------------------------------------------

    // -----------------------------------------------------------------------
    calculate_stride_reference(h_input, h_reference, memory_stride);
    float stride_transform_milliseconds = benchmark_kernel(
        stride_transform_kernel, threads_per_block, blocks_per_grid, warmup_iterations, repetitions,
        start, stop, d_input, d_output, problem_size, memory_stride);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    double stride_transform_bandwidth =
        calculate_effective_bandwidth(transform_useful_bytes, stride_transform_milliseconds);

    bool stride_valid = validate_output(h_output, h_reference);
    // --------------------------------------------------------------------

    // --------------------------------------------------------------------
    initialize_gather_indices(h_indices, input_size, gather_pattern);
    calculate_gather_reference(h_input, h_indices, h_reference);
    CHECK_CUDA(cudaMemcpy(d_indices, h_indices.data(), indices_bytes, cudaMemcpyHostToDevice));

    float gather_milliseconds =
        benchmark_kernel(gather_kernel, threads_per_block, blocks_per_grid, warmup_iterations,
                         repetitions, start, stop, d_input, d_output, d_indices, problem_size);

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));
    double gather_bandwidth =
        calculate_effective_bandwidth(gather_useful_bytes, gather_milliseconds);

    bool gather_valid = validate_output(h_output, h_reference);
    // -------------------------------------------------------------------------

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_indices));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    PRINT("Contiguous: {} ms, {} GB/s", scalar_transform_milliseconds, scalar_transform_bandwidth);
    PRINT("Offset {}: {} ms, {} GB/s", offset, offset_scalar_transform_milliseconds,
          offset_scalar_transform_bandwidth);
    PRINT("Memory stride {}: {} ms, {} GB/s", memory_stride, stride_transform_milliseconds,
          stride_transform_bandwidth);
    PRINT("Gather pattern {}: {} ms, {} GB/s", gather_pattern, gather_milliseconds,
          gather_bandwidth);

    PRINT("Validated {} outputs for the contiguous baseline, offset {}, memory stride {}, and "
          "gather pattern {} using {} blocks x {} threads.",
          problem_size, offset, memory_stride, gather_pattern, blocks_per_grid, threads_per_block);
    return transform_valid && offset_valid && stride_valid && gather_valid ? EXIT_SUCCESS
                                                                           : EXIT_FAILURE;
}

/*
Lab questions

1. What does coalescing mean at the warp level?

Coalescing means combining the memory addresses requested by the active lanes of a warp into as
few memory transactions as possible. For example, 32 neighboring lanes reading 32 neighboring
FP32 values request 128 useful bytes, which an aligned access can cover with four 32-byte sectors.
A strided or scattered access can require many more sectors for the same 128 useful bytes.

2. Why can aligned allocation still produce uncoalesced accesses?

Allocation alignment only aligns the base address. Coalescing depends on the addresses generated
by every lane for each memory instruction. An aligned array is still accessed inefficiently when
lanes use a large stride, scattered indices, or an offset that makes a warp cross extra sector
boundaries.

3. How do cache hits complicate a simple bandwidth interpretation?

Effective bandwidth counts the useful bytes requested by the algorithm, but it does not identify
where those bytes were served. Cache hits can satisfy inefficient requests without accessing DRAM,
so two kernels with the same useful bandwidth can generate different DRAM traffic. Conversely,
poor coalescing can fetch extra sectors whose data is unused. Cache hit rates, sector counts, and
DRAM traffic must therefore be considered together with effective bandwidth.

4. Which LLM operations naturally perform gather-like access?

Examples include embedding-table lookups from token IDs, Mixture-of-Experts token routing, paged
KV-cache access during attention, sparse attention, and selecting rows or tokens using index
arrays. Their data addresses depend on indices or routing decisions rather than neighboring thread
positions, so neighboring warp lanes may read unrelated memory locations.
*/
