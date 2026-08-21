# Lab 11: Shared-Memory Tiled GEMM

## Goal

Increase GEMM data reuse by cooperatively loading blocks of `A` and `B` into shared memory.

## Write

- Preserve the naive kernel as a baseline.
- A square shared-memory tiled FP32 GEMM.
- Bounds-safe loads that zero-fill partial tiles.
- Correct output stores for dimensions not divisible by tile size.
- At least two tile configurations with resource use recorded.

## Test Shapes

Reuse lab 10 tests and add dimensions one below, equal to, and one above every tile boundary. Include `M`, `N`, or `K` smaller than one tile and highly rectangular matrices.

## Benchmark

Compare naive, tiled, and cuBLAS kernels. Sweep shapes and tile sizes. Record shared-memory use, block size, achieved FLOP/s, and occupancy-related resource limits.

## Questions

1. How many output values reuse one loaded tile value?
2. Why must all participating threads reach block-wide barriers?
3. How do partial K tiles remain mathematically correct?
4. Why does a larger tile not guarantee better performance?

## Complete When

Partial tiles are correct, tiled GEMM measurably improves appropriate shapes, and the report relates tile choice to reuse and resource constraints.
