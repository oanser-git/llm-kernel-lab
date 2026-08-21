# Lecture: CUDA Execution And Hardware Mapping

Read this before implementing Lab 01. This lecture teaches the technical model; it does not provide a complete lab solution.

## 1. The Two Hierarchies

CUDA has a logical programming hierarchy and a physical hardware hierarchy.

```text
Programming model                 Hardware

Grid                              GPU
  -> Block                          -> Streaming Multiprocessor (SM)
      -> Thread                         -> Warp
                                               -> Lane
```

They are related, but they are not identical.

The program defines grids, blocks, and threads. The GPU scheduler assigns blocks to SMs. Each SM divides a resident block's threads into warps and executes those warps.

## 2. Logical CUDA Objects

### Grid

A grid is all work created by one kernel launch:

```cpp
kernel<<<number_of_blocks, threads_per_block>>>(arguments);
```

One launch creates one grid. Different kernel launches create different grids.

### Block

A block is a group of threads that can cooperate using:

- shared memory;
- block-wide barriers such as `__syncthreads()`;
- block-level collective operations.

Threads in different blocks cannot use `__syncthreads()` to synchronize with each other.

Blocks are intentionally independent so the scheduler can execute them in any order and on any available SM.

### Thread

A CUDA thread is one logical execution of the kernel function. Threads execute the same kernel program but observe different built-in indexes.

A thread index is not a permanent physical core number. It is an identifier that exists only for that launch.

## 3. Physical GPU Objects

### Streaming Multiprocessor

An SM is a major execution unit inside the GPU. Your GB10 reports 48 SMs.

An SM contains resources including:

- warp schedulers;
- execution pipelines;
- registers;
- shared memory;
- load/store hardware;
- special-function and matrix hardware.

A block is assigned to one SM and remains there until it completes. One SM can keep several blocks resident if threads, warps, registers, and shared-memory limits allow it.

### Warp

A warp is the hardware scheduling group for CUDA threads. Your GB10 has 32 lanes per warp.

```text
Warp 0: lanes 0-31
```

The SM issues instructions for warps, not for isolated CUDA threads one at a time. Threads in a warp normally follow the same instruction stream under the SIMT execution model.

### Lane

A lane is one thread position inside a warp. Lane IDs range from 0 through 31.

```text
lane_id = linear_thread_index_within_block % 32
```

A lane ID is reused in every warp. Lane 5 exists in warp 0, warp 1, and every later warp.

## 4. How Blocks Become Warps

For a one-dimensional block, CUDA forms warps from consecutive `threadIdx.x` values.

For 64 threads:

```text
Block with threadIdx.x 0-63

Warp 0: threads 0-31
Warp 1: threads 32-63
```

The number of warps required by a block is:

```text
warps_per_block = ceil(threads_per_block / warp_size)
```

Using integer arithmetic:

```text
warps_per_block = (threads_per_block + warp_size - 1) / warp_size
```

### Example: 50 Threads

```text
ceil(50 / 32) = 2 warps

Warp 0: 32 active lanes
Warp 1: 18 active lanes, 14 inactive lanes
```

Fifty threads are legal. The second warp is partially populated because hardware still schedules warp-sized groups.

### Example: 64 Threads

```text
64 / 32 = 2 warps

Warp 0: 32 active lanes
Warp 1: 32 active lanes
```

This is why common block sizes are multiples of 32. Multiples of 32 avoid partially populated warps inside full blocks. The final block of a problem can still contain threads that fail a bounds check.

## 5. GB10 Limits From Lab 00

| Property | GB10 value | Meaning for Lab 01 |
|---|---:|---|
| SM count | 48 | Enough blocks are needed to expose work across the GPU |
| Warp size | 32 | Block threads are grouped into sets of 32 lanes |
| Maximum threads per block | 1024 | A launch requesting more is invalid |
| Maximum threads per SM | 1536 | Resident blocks share this thread capacity |
| Maximum blocks per SM | 24 | Tiny blocks can reach this limit before the thread limit |
| Registers per SM | 65536 | Kernel register use can reduce resident warps or blocks |
| Shared memory per SM | 100 KiB | Blocks using shared memory divide this finite resource |

These are ceilings, not promises. A valid block size does not guarantee maximum performance.

For a simple indexing kernel with little resource use, block sizes such as 64, 128, or 256 are reasonable experiments. There is no universally best block size.

## 6. Launch Configuration

The CUDA launch syntax is:

```cpp
kernel<<<grid_dimension, block_dimension>>>(arguments);
```

For a one-dimensional launch:

```cpp
kernel<<<2, 64>>>(arguments);
```

This means:

```text
gridDim.x  = 2 blocks
blockDim.x = 64 threads per block
logical threads launched = 2 x 64 = 128
warps per block = 2
warps in the complete grid = 4
```

This does not mean block 0 runs before block 1. It does not mean thread 0 runs before thread 1. Scheduling order is intentionally unspecified.

## 7. Built-In Index Variables

For a one-dimensional launch, each thread can observe:

| Built-in value | Meaning |
|---|---|
| `threadIdx.x` | Thread index inside the current block |
| `blockIdx.x` | Block index inside the current grid |
| `blockDim.x` | Threads per block for this launch |
| `gridDim.x` | Blocks in the grid for this launch |
| `warpSize` | Hardware warp width, 32 on the GB10 |

These variables are supplied by CUDA. They are not ordinary variables initialized by your host code.

Indexes begin at zero:

```text
blockIdx.x  = 0, 1, 2, ...
threadIdx.x = 0, 1, 2, ...
```

Thread indexes restart in every block.

## 8. Linear Thread Identity

For a one-dimensional grid, a thread's unique linear position across blocks is:

```text
global_thread_index = block_index * threads_per_block + thread_index_in_block
```

Example with 64 threads per block:

```text
Block 0, thread 5 -> 0 * 64 + 5 = global thread 5
Block 1, thread 5 -> 1 * 64 + 5 = global thread 69
Block 2, thread 7 -> 2 * 64 + 7 = global thread 135
```

The global index is unique for the grid. Block, thread, warp, and lane indexes are local identifiers that repeat in later blocks or warps.

## 9. Warp And Lane Identity

For a one-dimensional block:

```text
warp_index_in_block = thread_index_in_block / warp_size
lane_index_in_warp  = thread_index_in_block % warp_size
```

Integer division finds which group of 32 contains the thread. Remainder finds the thread's position inside that group.

| `threadIdx.x` | Warp in block | Lane in warp |
|---:|---:|---:|
| 0 | 0 | 0 |
| 15 | 0 | 15 |
| 31 | 0 | 31 |
| 32 | 1 | 0 |
| 47 | 1 | 15 |
| 63 | 1 | 31 |

Warp numbering also restarts in each block.

## 10. Two-Dimensional And Three-Dimensional Blocks

CUDA permits `x`, `y`, and `z` dimensions. Hardware still forms warps using a linearized thread order, with `x` changing fastest.

```text
linear_thread_in_block =
    threadIdx.x
    + blockDim.x * threadIdx.y
    + blockDim.x * blockDim.y * threadIdx.z
```

Lane and warp identity must use this linear thread index for multidimensional blocks.

Lab 01 begins with one-dimensional launches. Do not add dimensions before the one-dimensional model is clear.

## 11. Problem Size Versus Launch Size

The problem size and the number of launched threads are different concepts.

```text
problem size:      elements that require work
launch size:       blocks x threads per block
```

If the problem contains 100 elements and blocks contain 64 threads:

```text
blocks required = ceil(100 / 64) = 2
threads launched = 2 x 64 = 128
valid element indexes = 0-99
extra thread indexes = 100-127
```

Launching extra threads is normal. Those threads must detect that they do not own valid data and stop before accessing memory.

This is why launch dimensions usually round upward instead of requiring every problem size to divide evenly by the block size.

## 12. Bounds And Memory Safety

An allocation for 100 integers has valid indexes 0 through 99. Accessing index 100 or greater is outside the allocation.

The safety rule is:

```text
calculate logical data index
compare it with the problem size
access memory only when the index is valid
```

A bounds check protects memory. It does not prevent the extra hardware lanes from being scheduled; those lanes simply stop before doing useful data work.

Use Compute Sanitizer later to detect invalid memory access:

```bash
compute-sanitizer --tool memcheck ./lab1
```

## 13. Host Code And Device Code

The same `.cu` file can contain two kinds of code:

```text
Host code
  -> ordinary C++ main function
  -> runs on the ARM CPU
  -> allocates memory and launches kernels

Device code
  -> __global__ kernel function
  -> runs as many CUDA threads on the GPU
  -> reads built-in thread and block indexes
```

The CPU does not become a GPU thread. The CPU submits a grid of work to the GPU.

## 14. Host And Device Memory

For this lab, the GPU produces index values and the CPU verifies them.

```text
CPU host vectors
  <- cudaMemcpy device-to-host
GPU device arrays
  <- written by the kernel
```

The conceptual lifecycle is:

```text
1. CPU determines problem and launch dimensions.
2. CPU allocates output storage on the GPU.
3. CPU launches the kernel.
4. GPU threads write their own output positions.
5. CPU waits for completion.
6. Results move from device memory to host memory.
7. CPU checks the mapping.
8. CPU releases GPU allocations.
```

No input array is required to discover indexes because CUDA supplies the built-in identifiers.

## 15. Why There Is No Write Race

A race occurs when threads access shared state without correct coordination and at least one access writes.

In this lab, every valid thread owns a different global output position:

```text
thread with global index 0 -> output position 0
thread with global index 1 -> output position 1
thread with global index 2 -> output position 2
```

If the global mapping is correct, two threads do not write the same location. No synchronization between threads is needed for these independent stores.

## 16. Asynchronous Kernel Launch

Kernel launch syntax normally returns control to the CPU before the GPU finishes:

```text
CPU launches kernel
CPU continues
GPU executes concurrently
```

This is asynchronous execution.

Two different checks are important during development:

```text
Launch error check
  -> Was the launch configuration valid?

Synchronization error check
  -> Did execution fail after launch?
```

A blocking device-to-host copy waits for earlier work in the same stream, but explicit synchronization makes the learning sequence easier to observe.

Host wall-clock timing around an asynchronous launch can measure only submission overhead unless the timing boundary waits for GPU completion.

## 17. Grid-Stride Loops

A one-element-per-thread mapping assumes the launch creates at least as many threads as elements.

A grid-stride loop allows one thread to process several positions separated by the total number of threads in the grid:

```text
first position = unique global thread index
stride = blocks in grid * threads per block
next position = current position + stride
```

Example with 128 launched threads and 1000 elements:

```text
thread 5 visits positions 5, 133, 261, 389, ...
thread 6 visits positions 6, 134, 262, 390, ...
```

This separates problem size from a fixed launch size and allows threads to perform repeated independent work.

Do not begin with this version. First understand the one-element-per-thread mapping, then extend it.

## 18. Divergence And Partial Warps

When lanes in a warp take different branches, the warp may need to execute paths with different lane masks. This is called divergence.

A final bounds check often creates one partially active final warp:

```text
lanes owning valid indexes -> continue
lanes beyond count         -> return
```

This small edge effect is normal. Choosing a non-multiple-of-32 block size creates a partial warp in every block, which can waste more lanes across a large grid.

## 19. Occupancy Is Not Utilization

Occupancy describes resident warps relative to an architectural maximum. It is affected by:

- block size;
- registers per thread;
- shared memory per block;
- maximum blocks per SM;
- maximum warps and threads per SM.

High occupancy does not guarantee high performance. Low occupancy does not always imply poor performance. It is one resource measurement, not a final score.

For Lab 01, focus on correct mapping and observable launch behavior. Performance interpretation comes after correctness.

## 20. Materialize The Model In Your Terminal

Revisit the GB10 facts produced by Lab 00:

```bash
cd ~/llm-kernel-lab/labs/00_device_query
./lab0
```

Confirm these values in its output:

```text
Warp size: 32
Maximum threads per block: 1024
Maximum threads per SM: 1536
Maximum blocks per SM: 24
Multiprocessors: 48
```

Inspect a real kernel's PTX and SASS from the stack walkthrough:

```bash
cd ~/llm-kernel-lab/walkthroughs/cuda_stack
less runtime_demo.ptx
cuobjdump --dump-sass runtime_demo.cubin | less
```

In PTX, find:

```text
%ctaid.x   block index
%tid.x     thread index
```

In SASS, find:

```text
SR_CTAID.X hardware special register for block ID
SR_TID.X   hardware special register for thread ID
```

This is the concrete path from CUDA built-in variables to hardware-visible special registers.

After writing your own Lab 01 kernel, inspect it the same way:

```bash
cuobjdump --dump-ptx ./lab1 | less
cuobjdump --dump-sass ./lab1 | less
```

## 21. Readiness Check

Do not begin the complete assignment until you can answer these without guessing:

1. What is the difference between a logical block and a physical SM?
2. Why does a 50-thread block require two warps?
3. Can a warp contain threads from different blocks?
4. Can a block move between SMs while executing?
5. Why is `threadIdx.x` not globally unique?
6. Why can a launch create more threads than the problem has elements?
7. What protects memory from extra threads?
8. Why can host timing around a kernel launch be misleading?
9. Why does a block size that is valid not necessarily perform well?
10. What is the difference between warp index and lane index?

## 22. Answer Key

1. A block is a logical group defined by the program; an SM is physical hardware that executes one or more resident blocks.
2. Hardware schedules groups of 32 lanes, so 50 threads occupy one full warp and one partial warp.
3. No. Every warp belongs to one block.
4. No. Once assigned, the block remains on that SM until completion.
5. Thread indexes restart at zero in every block.
6. Blocks have fixed integer dimensions, so launch size normally rounds upward to cover the problem.
7. A bounds check prevents out-of-range memory access.
8. Kernel launches are asynchronous, so host timing may measure submission instead of GPU completion.
9. Register use, shared memory, warp occupancy, memory access, and workload shape also matter.
10. Warp index selects a group of 32 threads; lane index selects a thread position inside that group.

## 23. What To Learn Before Coding

You are ready to start when this sentence makes sense:

```text
The CPU launches a grid of independent blocks; the scheduler assigns blocks to SMs; each block is divided into 32-lane warps; each thread computes a logical data index from CUDA-provided identifiers; and only valid threads access their assigned memory positions.
```
