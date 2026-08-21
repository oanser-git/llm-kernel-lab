# Lab 01: Kernel Launch And Indexing

## Before You Code

Read [`LECTURE.md`](LECTURE.md) completely before implementing the assignment. It explains the CUDA programming hierarchy, physical GB10 execution hierarchy, warp formation, partial warps, launch dimensions, built-in indexes, asynchronous execution, and the relevant hardware limits from Lab 00.

Use this order:

1. Read the lecture through the warp and lane sections.
2. Run the terminal observations in lecture section 20.
3. Complete the readiness check without reading its answer key first.
4. Begin the core mapping assignment.
5. Add grid-stride loops and timing only after the core mapping validates.

## Goal

Understand how grids, blocks, threads, and warps map a logical problem onto GPU execution.

## Core Assignment

- A kernel that stores each thread's global index, block index, thread index within its block, lane index within its warp, and warp index.
- A host launcher that accepts problem size and block size.
- Bounds checking for sizes not divisible by the block size.
- CPU validation for every written element.

## Extensions After The Core Works

- A grid-stride-loop version for inputs larger than the launched grid.
- Several legal block sizes, including one that is not a multiple of 32.
- One invalid launch configuration with a clearly reported CUDA error.
- CUDA-event timing compared with synchronized host timing.

## Test Shapes

Use `0`, `1`, `31`, `32`, `33`, `255`, `256`, `257`, and a size much larger than one grid. Test several legal block sizes and one invalid launch configuration.

## Benchmark

Measure tiny launches and enough work to occupy the GPU. Separate host-observed launch latency from CUDA-event kernel duration.

## Questions

1. Why does launching more threads than elements remain correct with a bounds check?
2. How is a warp different from a block?
3. Why can a grid-stride loop improve reuse of a fixed launch configuration?
4. Why is host wall-clock timing misleading without synchronization?

## Complete When

All mappings validate for awkward sizes, launch errors are reported, and the report explains asynchronous launch behavior.
