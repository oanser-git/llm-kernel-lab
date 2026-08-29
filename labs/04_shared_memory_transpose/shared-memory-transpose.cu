// Lab 04: Shared-Memory Transpose
// Read README.md, then implement the lab from scratch in this file.

#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>
#include <vector>

__device__ void print_tile_mapping(const float *__restrict tile, int tile_width, int tile_height,
                                   int rows, int columns, int block_id_x, int block_id_y,
                                   int thread_id_x, int thread_id_y)
{
    if (rows == 4 && columns == 4 && block_id_x == 1 && block_id_y == 0)
    {

        int input_column = block_id_x * tile_width + thread_id_x;
        int input_row = block_id_y * tile_height + thread_id_y;

        int shared_write_y = thread_id_y;
        int shared_write_x = thread_id_x;
        float shared_write_value = tile[shared_write_y * tile_width + shared_write_x];

        int shared_read_y = thread_id_x;
        int shared_read_x = thread_id_y;
        float shared_read_value = tile[shared_read_y * tile_width + shared_read_x];

        int output_column = block_id_y * tile_height + thread_id_x;
        int output_row = block_id_x * tile_width + thread_id_y;
        int output_index = output_row * rows + output_column;

        // Step 1: identify the block and thread. This can be tested immediately.
        printf("B(x=%d,y=%d) T(x=%d,y=%d)\n", block_id_x, block_id_y, thread_id_x, thread_id_y);

        // Step 2: show the global input coordinate handled by this thread.
        printf("  input: in(x=%d,y=%d)\n", input_column, input_row);

        // Step 3: show where that input value was written in shared memory.
        printf("  shared write: tile[y=%d][x=%d]=%.2f\n", shared_write_y, shared_write_x,
               shared_write_value);

        // Step 4: show the transposed shared position read by this thread.
        printf("  shared read: tile[y=%d][x=%d]=%.2f\n", shared_read_y, shared_read_x,
               shared_read_value);

        // Step 5: show the global output coordinate and flat output index.
        printf("  output: out(x=%d,y=%d), idx=%d\n\n", output_column, output_row, output_index);
    }
}

__global__ void copy_kernel(const float *__restrict__ input, float *__restrict__ output, int rows,
                            int columns)
{
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && column < columns)
    {
        int index = row * columns + column;
        output[index] = input[index];
    }
}

__global__ void naive_transpose_kernel(const float *__restrict__ input, float *__restrict__ output,
                                        int rows, int columns)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < columns)
    {
        output[c * rows + r] = input[r * columns + c];
    }
}

__global__ void tiled_transpose_kernel(const float *__restrict__ input, float *__restrict__ output,
                                       int rows, int columns)
{
    int tile_width = 32;
    int tile_height = 32;
    __shared__ float tile[32][32];

    int input_column = blockIdx.x * tile_width + threadIdx.x;
    int input_row = blockIdx.y * tile_height + threadIdx.y;

    if (input_column < columns && input_row < rows)
    {
        int idx = input_row * columns + input_column;
        tile[threadIdx.y][threadIdx.x] = input[idx];
    }

    __syncthreads();
    print_tile_mapping(&tile[0][0], tile_width, tile_height, rows, columns, blockIdx.x, blockIdx.y,
                       threadIdx.x, threadIdx.y);

    // in a warp of 32 threads, threadIdx.y stays the same, while threadIdx.x increments from 0
    // to 31 --> we use this
    int output_column = blockIdx.y * tile_height + threadIdx.x;
    int output_row = blockIdx.x * tile_width + threadIdx.y;

    if (output_column < rows && output_row < columns)
    {
        int idx = output_row * rows + output_column;
        output[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void padded_tiled_transpose_kernel(const float *__restrict__ input,
                                              float *__restrict__ output, int rows, int columns)
{
    int tile_width = 32;
    int tile_height = 32;
    __shared__ float tile[32][33];

    int input_column = blockIdx.x * tile_width + threadIdx.x;
    int input_row = blockIdx.y * tile_height + threadIdx.y;

    if (input_column < columns && input_row < rows)
    {
        int idx = input_row * columns + input_column;
        tile[threadIdx.y][threadIdx.x] = input[idx];
    }

    __syncthreads();

    // in a warp of 32 threads, threadIdx.y stays the same, while threadIdx.x increments from 0
    // to 31 --> we use this
    int output_column = blockIdx.y * tile_height + threadIdx.x;
    int output_row = blockIdx.x * tile_width + threadIdx.y;

    if (output_column < rows && output_row < columns)
    {
        int idx = output_row * rows + output_column;
        output[idx] = tile[threadIdx.x][threadIdx.y];
    }
}

void initialize_input(std::vector<float> &input)
{
    for (std::size_t index = 0; index < input.size(); ++index)
    {
        input[index] = static_cast<float>(index % 1009) * 0.25F;
    }
}

void calculate_copy_reference(const std::vector<float> &input, std::vector<float> &reference)
{
    reference = input;
}

bool validate_output(const std::vector<float> &output, const std::vector<float> &reference,
                     std::string_view kernel_name)
{
    for (std::size_t index = 0; index < output.size(); ++index)
    {
        if (output[index] != reference[index])
        {
            std::cerr << kernel_name << " validation failed at index " << index << ": expected "
                      << reference[index] << ", received " << output[index] << '\n';
            return false;
        }
    }

    return true;
}

template <typename KernelFunc, typename... Args>
float benchmark_kernel(KernelFunc kernel, dim3 grid, dim3 block, int warmup_iterations,
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

double calculate_effective_bandwidth(std::size_t element_count, float average_milliseconds)
{
    double useful_bytes =
        2.0 * static_cast<double>(element_count) * static_cast<double>(sizeof(float));
    double elapsed_seconds = static_cast<double>(average_milliseconds) / 1000.0;
    return useful_bytes / elapsed_seconds / 1.0e9;
}

void calculate_transpose_reference(std::vector<float> &in_mat, std::vector<float> &out_mat,
                                   int rows, int columns)
{
    int r = 0;
    int c = 0;
    int idx;
    int dst_idx;

    while (r < rows)
    {
        while (c < columns)
        {
            idx = r * columns + c;
            dst_idx = c * rows + r;
            out_mat[dst_idx] = in_mat[idx];
            c += 1;
        }
        c = 0;
        r += 1;
    }
}

void printMatrix(const std::vector<float> &matrix, int rows, int columns, std::string_view name,
                 const float *device_address = nullptr)
{
    if (rows > 8 || columns > 8)
    {
        return;
    }

    std::cout << "\n" << name << " (row-major memory order):\n";
    for (int row = 0; row < rows; ++row)
    {
        std::cout << "  ";
        for (int column = 0; column < columns; ++column)
        {
            std::size_t index = static_cast<std::size_t>(row) * columns + column;
            std::cout << std::fixed << std::setprecision(2) << matrix[index];
            if (device_address != nullptr)
            {
                std::cout << " (" << static_cast<const void *>(device_address + index) << ")";
            }
            if (column + 1 < columns)
            {
                std::cout << " -> ";
            }
        }
        std::cout << '\n';
        if (row + 1 < rows)
        {
            std::cout << "  |\n  v \n";
        }
    }
    std::cout << std::flush;
}

int main(int argc, char **argv)
{
    // Usage: ./shared-memory-transpose [rows] [columns] [warmups] [repetitions]
    int rows = argc >= 2 ? std::atoi(argv[1]) : 4096;
    int columns = argc >= 3 ? std::atoi(argv[2]) : 4096;
    int warmup_iterations = argc >= 4 ? std::atoi(argv[3]) : 10;
    int repetitions = argc >= 5 ? std::atoi(argv[4]) : 100;

    if (argc > 5 || rows <= 0 || columns <= 0 || warmup_iterations < 0 || repetitions <= 0)
    {
        std::cerr << "Usage: " << argv[0]
                  << " [positive_rows] [positive_columns] [nonnegative_warmups]"
                  << " [positive_repetitions]\n";
        return EXIT_FAILURE;
    }
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // We first need an answer that we trust. The CPU transpose gives us that answer without using
    // any CUDA idea. Every GPU version must match it, including rectangular matrices and edge
    // tiles. Only then does a speed comparison mean anything.
    std::size_t element_count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> h_in_mat(element_count);
    std::vector<float> h_out_mat(element_count);
    std::vector<float> h_copy_ref_mat(element_count);
    std::vector<float> h_ref_mat(element_count);
    initialize_input(h_in_mat);
    calculate_copy_reference(h_in_mat, h_copy_ref_mat);
    calculate_transpose_reference(h_in_mat, h_ref_mat, rows, columns);

    // Start with naive_transpose_kernel on purpose. Neighboring threads can walk across one input
    // row and read neighboring values. After the transpose, however, those same values belong to
    // one output column, so their output addresses are far apart. This version gives us a simple
    // result and lets us measure that global-memory problem before trying to fix it.
    dim3 threads_per_block(32, 32, 1);
    dim3 blocks_per_grid((columns + threads_per_block.x - 1) / threads_per_block.x,
                         (rows + threads_per_block.y - 1) / threads_per_block.y);

    float *d_in_mat = nullptr;
    float *d_out_mat = nullptr;

    std::size_t mat_bytes = element_count * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_in_mat, mat_bytes));
    CHECK_CUDA(cudaMalloc(&d_out_mat, mat_bytes));
    CHECK_CUDA(cudaMemcpy(d_in_mat, h_in_mat.data(), mat_bytes, cudaMemcpyHostToDevice));

    printMatrix(h_in_mat, rows, columns, "Input in device global memory", d_in_mat);
    // printMatrix(h_ref_mat, columns, rows, "CPU transpose reference");

    float copy_milliseconds =
        benchmark_kernel(copy_kernel, blocks_per_grid, threads_per_block, warmup_iterations,
                         repetitions, start, stop, d_in_mat, d_out_mat, rows, columns);
    CHECK_CUDA(cudaMemcpy(h_out_mat.data(), d_out_mat, mat_bytes, cudaMemcpyDeviceToHost));
    bool copy_valid = validate_output(h_out_mat, h_copy_ref_mat, "copy_kernel");

    float naive_transpose_milliseconds = benchmark_kernel(
        naive_transpose_kernel, blocks_per_grid, threads_per_block, warmup_iterations, repetitions,
        start, stop, d_in_mat, d_out_mat, rows, columns);

    CHECK_CUDA(cudaMemcpy(h_out_mat.data(), d_out_mat, mat_bytes, cudaMemcpyDeviceToHost));
    bool naive_transpose_valid = validate_output(h_out_mat, h_ref_mat, "naive_transpose_kernel");
    // printMatrix(h_out_mat, columns, rows, "Naive GPU output in device global memory", d_out_mat);

    // Now solve the read/write mismatch one tile at a time with tiled_transpose_kernel. Threads can
    // first read rows from global memory and place the values in a shared tile. After every value
    // in the tile is ready, the block can look at the same data in the transposed direction and
    // write output rows. Shared memory is the exchange area: the thread that writes a value does
    // not have to be the thread that loaded it. This is what allows both global operations to use
    // neighboring addresses. Keep this first tile unpadded so global coalescing and shared-memory
    // behavior can be observed separately.

    float tiled_transpose_milliseconds = benchmark_kernel(
        tiled_transpose_kernel, blocks_per_grid, threads_per_block, warmup_iterations, repetitions,
        start, stop, d_in_mat, d_out_mat, rows, columns);
    CHECK_CUDA(cudaMemcpy(h_out_mat.data(), d_out_mat, mat_bytes, cudaMemcpyDeviceToHost));
    bool tiled_transpose_valid = validate_output(h_out_mat, h_ref_mat, "tiled_transpose_kernel");
    // printMatrix(h_out_mat, columns, rows, "Tiled GPU output in device global memory", d_out_mat);

    // The tiled version fixes global access, but reading a column of a square shared tile can send
    // many lanes to the same bank. The padded_tiled_transpose_kernel tests that second problem.
    // Give each shared row one extra unused position so the next row starts in a different bank.
    // Keep the global work and the tile logic otherwise the same. Then a performance or counter
    // difference can be connected to bank mapping instead of some unrelated algorithm change.

    float padded_tiled_transpose_milliseconds = benchmark_kernel(
        padded_tiled_transpose_kernel, blocks_per_grid, threads_per_block, warmup_iterations,
        repetitions, start, stop, d_in_mat, d_out_mat, rows, columns);
    CHECK_CUDA(cudaMemcpy(h_out_mat.data(), d_out_mat, mat_bytes, cudaMemcpyDeviceToHost));
    bool padded_tiled_transpose_valid =
        validate_output(h_out_mat, h_ref_mat, "padded_tiled_transpose_kernel");

    // The four results tell one story. Copy shows the cost of moving the bytes without transposing.
    // Naive shows the cost of strided global access. Tiled shows what changes when global reads and
    // writes are coalesced. Padded tiled shows what changes when shared-bank conflicts are reduced.
    // Timing tells us how large each effect is, and NCU tells us whether sectors and bank-conflict
    // counters support that explanation.

    PRINT("Copy: {} ms, {} GB/s", copy_milliseconds,
          calculate_effective_bandwidth(element_count, copy_milliseconds));
    PRINT("Naive transpose: {} ms, {} GB/s", naive_transpose_milliseconds,
          calculate_effective_bandwidth(element_count, naive_transpose_milliseconds));
    PRINT("Tiled transpose: {} ms, {} GB/s", tiled_transpose_milliseconds,
          calculate_effective_bandwidth(element_count, tiled_transpose_milliseconds));
    PRINT("Padded tiled transpose: {} ms, {} GB/s", padded_tiled_transpose_milliseconds,
          calculate_effective_bandwidth(element_count, padded_tiled_transpose_milliseconds));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_in_mat));
    CUDA_CHECK(cudaFree(d_out_mat));

    bool all_valid = copy_valid && naive_transpose_valid && tiled_transpose_valid &&
                     padded_tiled_transpose_valid;
    PRINT("Validated {} x {} matrix: {}", rows, columns, all_valid ? "PASS" : "FAIL");
    return all_valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
