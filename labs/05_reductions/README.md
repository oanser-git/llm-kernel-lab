# Lab 05: Sum And Max Reductions

## Goal

Learn how threads cooperate to reduce many values while controlling synchronization, memory traffic, and floating-point behavior.

## Write

- FP32 CPU references for sum and maximum.
- A deliberately simple GPU baseline.
- A block-level shared-memory reduction.
- A warp-aware version using shuffle operations after you understand the shared-memory version.
- A multi-block strategy for inputs larger than one block can process.
- An explicit policy for empty input and NaN values.

## Test Shapes

Include `0`, `1`, warp boundaries, block boundaries, non-powers of two, large arrays, all-negative maximum inputs, repeated maxima, large dynamic range, NaN, and infinity.

## Benchmark

Sweep reduction length and block size. Distinguish kernel-only time from allocation of intermediate storage. Compare sum error across versions and input orderings.

## Questions

1. Which synchronization scopes are required at each reduction stage?
2. Why is a power-of-two-only implementation incomplete?
3. Why can parallel sum differ from sequential sum?
4. When does a two-kernel reduction become preferable to excessive atomic contention?

## Complete When

Sum and max work for arbitrary lengths, numerical behavior is documented, and profiler evidence identifies the dominant stalls in at least two versions.
