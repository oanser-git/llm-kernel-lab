# Lab 17: KV-Cache Layout

## Goal

Design, populate, and read key/value storage for autoregressive inference while understanding how layout changes the next-token access pattern.

## Write

- A precise logical-to-physical indexing formula.
- Append kernels for one token and multiple prompt tokens.
- Read or gather kernels that reconstruct logical tensors for validation.
- At least two layouts, such as token-major and head-major, for measured comparison.
- Per-sequence lengths with capacity checks and no out-of-bounds writes.
- Optional page-table indirection only after contiguous layouts are understood.

## Test Shapes

Include multiple batches, KV heads, head dimensions, capacities, prompt lengths, append positions, zero-length histories, full capacity, and attempted overflow. Use unique values that make indexing mistakes obvious.

## Benchmark

Measure append and full-history read independently. Report address patterns, effective bandwidth, and which layout favors decode attention versus bulk prompt writes.

## Questions

1. Which dimension should neighboring lanes traverse during decode?
2. Why do variable sequence lengths complicate batching?
3. What fragmentation problem does paging address?
4. Which metadata must be read before a paged cache address is known?

## Complete When

Two layouts round-trip correctly, capacity errors are detected, and the report uses measured access behavior to select a layout for decode.
