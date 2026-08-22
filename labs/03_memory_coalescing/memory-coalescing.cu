// Lab 03: Memory Coalescing
// Read README.md, then implement the lab from scratch in this file.
#include <cuda_runtime.h>

#include "../../common/cuda_utils.cuh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <cstdio>

__global__ void scalar_transform_kernel(const float *__restrict__ in, float *__restrict__ out,
                                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < n)
    {
        // Contiguous read, simple transform, contiguous write
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

__global__ void stride_transform_kernel(const float *__restrict__ in, float *__restrict__ out,
                                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < n; i += stride)
    {
        // Contiguous read, simple transform, contiguous write
        out[i] = in[i] * 2.0f + 1.0f;
    }
}

int main()
{

    return EXIT_SUCCESS;
}
