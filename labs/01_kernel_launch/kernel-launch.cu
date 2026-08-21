// Lab 01: Kernel Launch And Indexing - student version

#include <cuda_runtime.h>
#include <iostream>

// threadIdx.x 	Thread index inside the current block
// blockIdx.x	Block index inside the current grid
// blockDim.x	Threads per block for this launch
// gridDim.x	Blocks in the grid for this launch
// warpSize	    Hardware warp width, 32 on the GB10

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

__global__ void lab1(float *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        d_arr[idx] = idx;
    }
}

int main(int argc, char **argv)
{

    int N = 1000;
    size_t bytes = N * sizeof(float);

    float *h_arr = new float[N];

    // for (int i = 0; i < N; ++i)
    // h_arr[i] = static_cast<float>(i);

    float *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, bytes));

    cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    lab1<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    //
    // Copy resslt back from Device to Host
    cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    delete[] h_arr;

    return EXIT_SUCCESS;
}
