# Lab 21: CUDA Streams And Overlap

## Goal

Understand ordering, dependencies, and overlap by composing already-correct kernels on multiple CUDA streams.

First perform isolated experiments in this lab. Then integrate only measured, useful overlap into the working engine from Lab 20.

## Write

- A single-stream baseline containing at least two independent operations.
- A multi-stream version with independent input/output storage.
- CUDA events that express required cross-stream dependencies without global synchronization.
- Asynchronous memory operations appropriate to the allocation type used.
- NVTX ranges or clear labels for Nsight Systems.
- A deliberately incorrect dependency experiment that is explained and then removed or guarded from normal execution.

## Experiments

Compare tiny kernels, substantial kernels, independent compute, dependent compute, and any supported copy/compute overlap. Repeat while the GPU is otherwise idle.

## Questions

1. What ordering is guaranteed within one stream?
2. What ordering exists between streams without events?
3. Which hardware resources must be available for actual overlap?
4. Why can adding streams make a workload slower?
5. How does unified memory change, but not eliminate, data-movement concerns?

## Complete When

The Nsight Systems timeline demonstrates understood ordering, any claimed overlap is visible rather than inferred, and all dependencies are explicit.
