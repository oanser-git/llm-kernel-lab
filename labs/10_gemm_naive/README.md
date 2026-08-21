# Lab 10: Naive GEMM

## Goal

Implement matrix multiplication directly and establish a correct, deliberately simple baseline before introducing tiling.

## Contract

For row-major matrices:

```text
A: [M, K]
B: [K, N]
C: [M, N]
C[m, n] = sum_k A[m, k] * B[k, n]
```

Start with FP32 input, accumulation, and output. Document whether dimensions use tightly packed or explicit leading strides.

## Write

- Triple-loop CPU reference.
- One-output-element-per-thread CUDA kernel that reads directly from global memory.
- Correct launch geometry and boundary handling for arbitrary positive dimensions.
- Timing that excludes allocation and host-device transfers.
- cuBLAS SGEMM comparison with layouts verified rather than guessed.

## Test Shapes

Include tiny matrices that can be inspected manually, rectangular shapes, dimensions not divisible by block dimensions, `K=1`, realistic projection-like shapes, zeros, and random signed values.

## Benchmark

Report latency, calculated FLOP/s, arithmetic intensity assumptions, and relative performance against cuBLAS. Sweep at least square and highly rectangular cases.

## Questions

1. How many times does the naive kernel reread each useful input value?
2. Which matrix has a naturally coalesced access under your thread mapping?
3. Why is the theoretical operation count approximately `2*M*N*K`?
4. Why can a correct row-major cuBLAS comparison require transposition or operand-order care?

## Complete When

Arbitrary dimensions match the reference, layout is explicit, and poor baseline performance is explained using reuse and memory traffic.
