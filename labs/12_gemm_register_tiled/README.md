# Lab 12: Register-Tiled GEMM

## Goal

Make each thread compute multiple output values, increasing reuse from shared memory while managing register pressure and instruction-level parallelism.

## Write

- Preserve the shared-memory tiled baseline.
- A two-dimensional thread tile held in registers.
- Cooperative global-to-shared loads distinct from the output thread mapping.
- Vectorized loads only where alignment and tails are proven safe.
- Optional double buffering or asynchronous copy after the basic register-tiled kernel is understood.

## Test Shapes

Use all prior correctness shapes, especially dimensions that violate vector width, block tile size, thread tile size, and K-step assumptions.

## Benchmark

Sweep block tile, thread tile, K tile, and shape. Record registers per thread, shared memory per block, spills, active warps, achieved FLOP/s, and comparison with cuBLAS.

## Questions

1. What reuse is gained by computing several outputs per thread?
2. When does register pressure reduce useful concurrency?
3. How would a register spill appear in memory and profiler metrics?
4. Which dependency chains limit instruction-level parallelism?

## Complete When

The register-tiled kernel is correct for edge shapes and its performance is explained using measured register, occupancy, and memory behavior.
