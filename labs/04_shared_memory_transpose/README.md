# Lab 04: Shared-Memory Transpose

## Goal

Use tiling to turn an unfavorable global-memory access pattern into coalesced reads and writes, then identify shared-memory bank conflicts.

## Contract

Transpose a row-major `rows x columns` FP32 matrix into a row-major `columns x rows` output. Rectangular and partial edge tiles must work.

## Write

- CPU reference.
- Naive out-of-place transpose.
- Shared-memory tiled transpose.
- A padded shared-memory tile designed to reduce bank conflicts.
- A direct copy baseline for context.

## Test Shapes

Include square, rectangular, `1 x N`, dimensions smaller than a tile, dimensions one above and below tile boundaries, and dimensions not divisible by the tile size.

## Benchmark

Report effective bandwidth for copy, naive transpose, tiled transpose, and padded tiled transpose. Inspect global-memory transactions and shared-memory bank conflicts.

## Questions

1. Why can the naive kernel coalesce either reads or writes but not both?
2. Why does shared memory help?
3. Why can a square shared-memory tile introduce bank conflicts?
4. What does padding change in the address-to-bank mapping?

## Complete When

All rectangular cases are correct and profiler evidence explains both the global-access improvement and any bank-conflict change.
