// Lab 05: Sum And Max Reductions
// Read README.md, then implement the lab from scratch in this file.

#include <cuda_runtime.h>
#include <math_constants.h>

#include "../../common/cuda_utils.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

void init_input(std::vector<float> &input, int input_len)
{
    for (int index = 0; index < input_len; ++index)
    {
        input[index] = static_cast<float>((index % 101) - 50) * 0.25F;
    }
}

void calculate_sum_reference(const std::vector<float> &input, float &output, int input_len)
{
    output = 0.0F;

    for (int i = 0; i < input_len; ++i)
    {
        if (std::isnan(input[i]))
        {
            output = std::numeric_limits<float>::quiet_NaN();
            return;
        }

        output += input[i];
    }
}

void calculate_max_reference(const std::vector<float> &input, float &output, int input_len)
{
    output = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < input_len; ++i)
    {
        if (std::isnan(input[i]))
        {
            output = std::numeric_limits<float>::quiet_NaN();
            return;
        }

        if (input[i] > output)
        {
            output = input[i];
        }
    }
}

bool reference_value_matches(float actual, float expected)
{
    return std::isnan(expected) ? std::isnan(actual) : actual == expected;
}

bool run_reference_test(std::string_view name, const std::vector<float> &input, float expected_sum,
                        float expected_max)
{
    float actual_sum = 0.0F;
    float actual_max = 0.0F;
    int input_len = static_cast<int>(input.size());

    calculate_sum_reference(input, actual_sum, input_len);
    calculate_max_reference(input, actual_max, input_len);

    bool sum_valid = reference_value_matches(actual_sum, expected_sum);
    bool max_valid = reference_value_matches(actual_max, expected_max);

    PRINT("{}: sum={}, max={} [{}]", name, actual_sum, actual_max,
          sum_valid && max_valid ? "PASS" : "FAIL");
    return sum_valid && max_valid;
}

__global__ void baseline_sum_kernel(const float *__restrict__ input, int input_len, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < input_len)
    {
        float val = input[idx];
        if (!::isnan(val))
        {
            atomicAdd(output, val);
        }
        else
        {
            *output = 0.0f / 0.0f;
        }
    }
}
__global__ void baseline_max_kernel(const float *__restrict__ input, int input_len, float *output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        float result = -CUDART_INF_F;
        for (int index = 0; index < input_len; ++index)
        {
            float value = input[index];
            if (::isnan(value))
            {
                *output = CUDART_NAN_F;
                return;
            }

            if (value > result)
            {
                result = value;
            }
        }
        *output = result;
    }
}

template <typename KernelFunc, typename... Args>
float benchmark_kernel(KernelFunc kernel, int grid, int block, int warmup_iterations,
                       int repetitions, cudaEvent_t start, cudaEvent_t stop, Args... args)
{
    for (int iteration = 0; iteration < warmup_iterations; ++iteration)
    {
        kernel<<<grid, block>>>(args...);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    for (int iteration = 0; iteration < repetitions; ++iteration)
    {
        kernel<<<grid, block>>>(args...);
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&total_milliseconds, start, stop));
    return total_milliseconds / static_cast<float>(repetitions);
}

int main(int argc, char **argv)
{
    // Usage: ./reductions [problem_size] [threads_per_block] [warmups] [repetitions]
    int problem_size = argc >= 2 ? std::atoi(argv[1]) : 10000000;
    int threads_per_block = argc >= 3 ? std::atoi(argv[2]) : 256;
    int warmup_iterations = argc >= 4 ? std::atoi(argv[3]) : 10;
    int repetitions = argc >= 5 ? std::atoi(argv[4]) : 100;

    if (argc > 5 || problem_size < 0 || threads_per_block <= 0 || warmup_iterations < 0 ||
        repetitions <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [nonnegative_problem_size] [positive_threads_per_block]"
                  << " [nonnegative_warmups] [positive_repetitions]\n";
        return EXIT_FAILURE;
    }

    // CHECKPOINT 1: Define the result before writing a kernel.
    // A reduction turns many values into one value, so edge cases become part of the operation.
    // Decide what sum and max return for an empty input and what happens when any input is NaN.
    // Use the same policy in the CPU reference, every GPU version, and validation.

    // sum([1, 2, 3])       = 6
    // sum([])              = 0
    // sum([1, NaN, 3])     = NaN

    // max([1, 7, 3])       = 7
    // max([])              = -infinity
    // max([1, NaN, 3])     = NaN

    // CHECKPOINT 2: Write calculate_sum_reference and calculate_max_reference above main.
    // These are the answers you trust. Include all-negative data so max cannot pass by starting
    // from zero, and remember that a sum can change when the order of additions changes.

    float nan = std::numeric_limits<float>::quiet_NaN();
    float infinity = std::numeric_limits<float>::infinity();

    bool reference_tests_valid = true;
    reference_tests_valid &= run_reference_test("empty", {}, 0.0F, -infinity);
    reference_tests_valid &= run_reference_test("one value", {5.0F}, 5.0F, 5.0F);
    reference_tests_valid &=
        run_reference_test("all negative", {-8.0F, -2.0F, -5.0F}, -15.0F, -2.0F);
    reference_tests_valid &=
        run_reference_test("repeated max", {2.0F, 7.0F, 7.0F, 3.0F}, 19.0F, 7.0F);
    reference_tests_valid &= run_reference_test("NaN", {1.0F, nan, 3.0F}, nan, nan);
    reference_tests_valid &=
        run_reference_test("infinity", {1.0F, infinity, 3.0F}, infinity, infinity);
    reference_tests_valid &=
        run_reference_test("large first", {1.0e20F, 1.0F, -1.0e20F}, 0.0F, 1.0e20F);
    reference_tests_valid &=
        run_reference_test("cancellation first", {1.0e20F, -1.0e20F, 1.0F}, 1.0F, 1.0e20F);

    if (!reference_tests_valid)
    {
        return EXIT_FAILURE;
    }

    std::vector<float> h_input(problem_size);
    init_input(h_input, problem_size);

    float cpu_sum_reference = 0.0F;
    float cpu_max_reference = -std::numeric_limits<float>::infinity();
    calculate_sum_reference(h_input, cpu_sum_reference, problem_size);
    calculate_max_reference(h_input, cpu_max_reference, problem_size);

    float sum_identity = 0.0F;
    float max_identity = -std::numeric_limits<float>::infinity();
    float gpu_sum_result = sum_identity;
    float gpu_max_result = max_identity;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // CHECKPOINT 3: Write baseline_sum_kernel and baseline_max_kernel above main.
    // Start with the simplest correct GPU idea, even if it serializes or creates contention. This
    // baseline gives the later cooperative versions a clear problem to solve and measure.
    //
    std::size_t input_bytes = static_cast<std::size_t>(problem_size) * sizeof(float);
    std::size_t scalar_bytes = sizeof(float);

    float *d_input = nullptr;
    float *d_max_result = nullptr;
    float *d_sum_result = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_max_result, scalar_bytes));
    CHECK_CUDA(cudaMalloc(&d_sum_result, scalar_bytes));
    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_max_result, &max_identity, scalar_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_sum_result, &sum_identity, scalar_bytes, cudaMemcpyHostToDevice));

    int blocks_per_grid = (problem_size - 1) / threads_per_block + 1;

    // Validate one independent reduction before timing repeated launches.
    baseline_sum_kernel<<<blocks_per_grid, threads_per_block>>>(d_input, problem_size,
                                                                d_sum_result);
    baseline_max_kernel<<<1, 1>>>(d_input, problem_size, d_max_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CHECK_CUDA(
        cudaMemcpy(&gpu_sum_result, d_sum_result, scalar_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(
        cudaMemcpy(&gpu_max_result, d_max_result, scalar_bytes, cudaMemcpyDeviceToHost));

    bool baseline_sum_valid = reference_value_matches(gpu_sum_result, cpu_sum_reference);
    bool baseline_max_valid = reference_value_matches(gpu_max_result, cpu_max_reference);

    PRINT("Baseline sum: GPU={}, CPU={} [{}]", gpu_sum_result, cpu_sum_reference,
          baseline_sum_valid ? "PASS" : "FAIL");
    PRINT("Baseline max: GPU={}, CPU={} [{}]", gpu_max_result, cpu_max_reference,
          baseline_max_valid ? "PASS" : "FAIL");

    // Reset once before timing. The timed atomic-sum launches accumulate, so their final output is
    // not used for validation; correctness was checked with the independent launch above.
    CHECK_CUDA(
        cudaMemcpy(d_sum_result, &sum_identity, scalar_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(
        cudaMemcpy(d_max_result, &max_identity, scalar_bytes, cudaMemcpyHostToDevice));

    float baseline_sum_milliseconds = benchmark_kernel(
        baseline_sum_kernel, blocks_per_grid, threads_per_block, warmup_iterations, repetitions,
        start, stop, d_input, problem_size, d_sum_result);
    float baseline_max_milliseconds =
        benchmark_kernel(baseline_max_kernel, 1, 1, warmup_iterations, repetitions, start, stop,
                         d_input, problem_size, d_max_result);

    PRINT("Baseline sum time: {} ms", baseline_sum_milliseconds);
    PRINT("Baseline max time: {} ms", baseline_max_milliseconds);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_sum_result));
    CUDA_CHECK(cudaFree(d_max_result));

    if (!baseline_sum_valid || !baseline_max_valid)
    {
        return EXIT_FAILURE;
    }

    // CHECKPOINT 4: Write shared_sum_kernel and shared_max_kernel above main.
    // Give one block a chunk of input. Threads first contribute values to one shared workspace,
    // then repeatedly combine partial results until the block owns one answer. A
    // synchronization point is needed whenever one stage reads values produced by other threads
    // in the prior stage.

    // CHECKPOINT 5: Make the shared reduction work for arbitrary lengths.
    // Real inputs and final blocks are not always powers of two or full. A thread with no
    // input, or a reduction step with no partner, must still preserve the operation's result
    // instead of reading outside the array or silently dropping a value.

    // CHECKPOINT 6: Complete the reduction across multiple blocks.
    // Shared memory and __syncthreads() stop at the block boundary. Let each block produce one
    // partial result, then decide how those partials become one final result without pretending
    // that an ordinary grid-wide barrier exists inside the first kernel.

    // CHECKPOINT 7: Only after the shared version is clear, write warp_sum_kernel and
    // warp_max_kernel. Threads in one warp can exchange register values with shuffle
    // operations. This can remove some shared-memory traffic and block barriers, but the
    // cooperation scope is only one warp, so a separate plan is still needed to combine warps
    // and blocks.

    // CHECKPOINT 8: Validate behavior, not just ordinary positive inputs.
    // Cover zero and one element, warp and block boundaries, non-powers of two, all-negative
    // max, repeated maxima, large dynamic range, NaN, infinity, and a large multi-block input.
    // Compare max according to the chosen policy and measure sum error rather than assuming
    // bitwise equality.

    // CHECKPOINT 9: Benchmark only after every version is correct.
    // Keep allocation and intermediate-storage setup outside kernel-only timing. Sweep input
    // length and block size, use warm-ups and repeated CUDA-event measurements, and report
    // which bytes are useful algorithm data versus extra partial-result traffic introduced by
    // an implementation.

    // CHECKPOINT 10: Use NCU to explain the versions, not to replace timing.
    // Compare at least the simple and cooperative kernels. Look for the cost of contention,
    // synchronization, memory traffic, occupancy, and stalls, then connect the counters to the
    // cooperation strategy used by each version.

    return EXIT_SUCCESS;
}
