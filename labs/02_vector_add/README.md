# Lab 02: Vector Addition

## Before You Code

Read [`LECTURE.md`](LECTURE.md) before implementing the lab. Lab 01 taught launch geometry and indexing; this lecture adds buffer ownership, transfer directions, contiguous memory access, arithmetic intensity, memory-bound behavior, CUDA events, bandwidth calculation, floating-point edge cases, grid-stride loops, and vectorization safety.

Use this order:

1. Read through the memory-traffic and arithmetic-intensity sections.
2. Complete the terminal observations and readiness check.
3. Implement the CPU reference and scalar CUDA baseline.
4. Validate all boundary shapes before timing.
5. Add the grid-stride version.
6. Add CUDA-event timing and bandwidth calculations.
7. Attempt vectorized loads only as an optional final extension.

## Goal

Build the first complete data-parallel kernel and measure memory bandwidth correctly.

## Contract

For contiguous FP32 vectors of length `n`:

```text
out[i] = a[i] + b[i]
```

## Stage 1: Correct Scalar Baseline

- CPU reference, GPU allocation, initialization, transfer, kernel launch, and validation.
- A scalar CUDA baseline with a bounds check.

## Stage 2: Grid-Stride Version

- A grid-stride-loop variant using the same mathematical contract.
- Correctness comparison between scalar and grid-stride outputs.

## Stage 3: Measurement

- Warm-up launches before measurement.
- CUDA-event timing that excludes allocation, initialization, and transfer.
- Effective bandwidth using a documented `12 * N` useful-byte model.
- At least two launch configurations.
- A device-to-device copy baseline with its own documented byte model.

## Stage 4: Optional Vectorization

- An aligned vectorized-load variant after scalar and grid-stride paths are correct.
- Explicit alignment checks or assumptions.
- Correct tail handling when `N` is not a multiple of the vector width.

## Test Shapes

Include zero length, one element, values around block boundaries, large vectors, misaligned offsets for scalar code, NaN, infinity, and random finite values.

## Benchmark

Report latency and effective bandwidth. State exactly how many useful bytes per element you count. Compare against a device-to-device copy baseline rather than a marketing bandwidth number alone.

## Questions

1. Why is vector addition normally memory-bound?
2. When is vectorization unsafe?
3. Why can a grid-stride loop and a one-element-per-thread kernel perform similarly?
4. Which timing components belong in an LLM operator benchmark?

## Complete When

The scalar implementation is correct, bandwidth calculation is justified, and at least two launch configurations are compared.
