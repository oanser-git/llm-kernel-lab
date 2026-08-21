# Lecture: Vector Addition, Memory Traffic, And GPU Timing

Read this before implementing Lab 02. It teaches the technical model and measurement method without providing a complete vector-add program.

## 1. What Changes After Lab 01

Lab 01 generated metadata from CUDA's built-in indexes. Lab 02 applies the same indexing model to real input data:

```text
out[i] = a[i] + b[i]
```

Every valid logical element requires:

```text
two input values
one floating-point addition
one output value
```

The launch hierarchy is not new. The new topics are data movement, memory access patterns, floating-point validation, asynchronous timing, and bandwidth analysis.

## 2. Three Separate Vectors

Vector addition uses three logical buffers:

```text
a       input, read only
b       input, read only
output  result, written by the kernel
```

For length `N`, each FP32 vector occupies:

```text
N * sizeof(float)
= N * 4 bytes
```

All three vectors together occupy approximately:

```text
3 * N * 4 bytes
= 12 * N bytes
```

This is storage capacity, not yet a claim about measured memory traffic.

## 3. Host And Device Ownership

The CPU prepares inputs and validates outputs. The GPU kernel processes device-accessible buffers.

```text
CPU host memory                         GPU-accessible memory

h_a  ---------------- H2D -----------> d_a
h_b  ---------------- H2D -----------> d_b
h_out <--------------- D2H ------------ d_out
```

The transfer directions are:

| Direction | Meaning | Vector-add use |
|---|---|---|
| H2D | Host to Device | Copy `a` and `b` before the kernel |
| D2H | Device to Host | Copy `output` back for validation |
| D2D | Device to Device | Useful later as a bandwidth baseline |

The output is produced by the GPU. Copying an expected output to `d_out` before launch can hide a broken kernel and create a false-positive test.

## 4. Unified Memory Does Not Remove The Model

The phrase "unified memory" is overloaded. Three different ideas must not be confused:

| Idea | What it means | What it does not guarantee |
|---|---|---|
| Unified physical memory | CPU and GPU use one physical memory pool on the GB10 | Every allocation is automatically the same buffer or equally fast from both processors |
| Unified Virtual Addressing (UVA) | CUDA can represent host and device addresses in one virtual address model | Every pointer can be dereferenced safely by both CPU and GPU |
| CUDA Unified Memory | `cudaMallocManaged` creates an allocation managed for CPU/GPU access | No synchronization, placement cost, cache behavior, or bandwidth limit |

### One Physical Pool Does Not Mean One Logical Buffer

Consider two allocations:

```text
float *h_a = CPU allocation
float *d_a = cudaMalloc allocation
```

On a discrete PCIe GPU, they are normally backed by different physical memory systems: CPU DRAM and GPU VRAM.

On the GB10, both may ultimately use the same unified physical memory pool. They are still two separate allocations with different addresses and CUDA access semantics:

```text
h_a does not automatically alias d_a
writing h_a does not mean d_a contains the same values
cudaMalloc memory should be used according to CUDA device-allocation rules
```

An explicit `cudaMemcpy` copies values from one logical allocation to another. On the GB10 this transfer may not travel over a PCIe link to separate VRAM, but it still performs ordered memory work, consumes bandwidth, and takes time.

### Shared Memory Hardware Does Not Remove Synchronization

The CPU and GPU execute independently:

```text
CPU submits a kernel
CPU may continue immediately
GPU reads or writes memory later
```

If the GPU writes an output, the CPU must not consume that result before GPU completion. A synchronization operation or a blocking copy establishes the required ordering.

Even when hardware supports coherent access, coherence and synchronization answer different questions:

```text
coherence:       how caches obtain a consistent value
synchronization: when one processor is allowed to rely on another processor's completed work
```

Coherent hardware does not make a data race correct.

### Managed Memory Removes Some Copies, Not The Memory Model

A managed allocation can be accessed by CPU and GPU using one pointer:

```text
CPU writes managed allocation
GPU kernel accesses managed allocation
CPU waits for GPU completion
CPU reads result
```

The explicit H2D and D2H copies may disappear, but these concerns remain:

- CPU/GPU execution ordering;
- allocation lifetime;
- page placement or migration policy;
- CPU and GPU cache behavior;
- memory bandwidth and latency;
- contention with other processors and applications;
- whether the access pattern is efficient for GPU warps.

### The GB10 Still Has A Memory Hierarchy

Unified physical memory is not a single-cycle register file. Data still travels through a hierarchy such as:

```text
CPU cores and CPU caches
          |
unified physical memory pool
          |
GPU L2, GPU caches, SM load/store units, registers
```

The CPU and GPU use different execution pipelines and cache paths. A GPU kernel can still be memory-bound because the shared pool has finite bandwidth. CPU activity can also contend for that bandwidth.

### Why Lab 02 Uses Explicit Allocations And Copies

Lab 02 deliberately uses:

```text
host input allocations
cudaMalloc device allocations
H2D copies for inputs
kernel execution
D2H copy for output validation
```

This makes every responsibility visible:

```text
where input values originate
which buffers the kernel accesses
when GPU work begins and ends
which operations belong outside kernel-only timing
when the CPU is allowed to validate output
```

After this explicit baseline is correct and measured, a later experiment can replace the allocations with `cudaMallocManaged` and compare behavior. Starting with managed memory would hide distinctions that this lab is intended to teach.

## 5. One Thread Per Element

The scalar baseline assigns one logical element to each valid thread:

```text
global thread 0 -> output element 0
global thread 1 -> output element 1
global thread 2 -> output element 2
...
```

The mapping from Lab 01 remains:

```text
global index = block index * threads per block + thread index in block
```

The bounds rule also remains:

```text
if global index is outside [0, N), do not access the vectors
```

The new behavior at a valid position is conceptually:

```text
read a[index]
read b[index]
add the two FP32 values
write output[index]
```

## 6. Why Contiguous Access Matters

Consider one warp whose lanes own consecutive indexes:

```text
lane 0  -> a[0]
lane 1  -> a[1]
lane 2  -> a[2]
...
lane 31 -> a[31]
```

Each FP32 value occupies 4 bytes, so the warp requests 32 consecutive values:

```text
32 lanes * 4 bytes = 128 useful bytes
```

Neighboring lanes accessing neighboring addresses allows the memory subsystem to combine requests into efficient transactions. This is called coalescing.

If lanes instead access widely separated addresses, the GPU may need more memory transactions to deliver the same useful data. Lab 03 studies this in detail; Lab 02 establishes the contiguous baseline.

## 7. Useful Memory Traffic Per Element

Ignoring cache effects and implementation overhead, one scalar vector-add element performs:

```text
read a[i]      4 bytes
read b[i]      4 bytes
write out[i]   4 bytes
----------------------
useful traffic 12 bytes
```

For `N` elements:

```text
useful bytes = 12 * N
```

This is a useful-byte model. Actual hardware transactions can differ because of cache lines, sector requests, write behavior, alignment, partial warps, and cache hits.

Always state which byte model a bandwidth number uses.

## 8. Arithmetic Intensity

Arithmetic intensity compares useful arithmetic operations with bytes moved:

```text
arithmetic intensity = operations / bytes
```

Vector addition performs one FP32 addition for approximately 12 useful bytes:

```text
1 FLOP / 12 bytes
= 0.083 FLOP per byte
```

This is very low arithmetic intensity. The GPU spends much more effort moving data than performing arithmetic.

## 9. Why Vector Addition Is Memory-Bound

A kernel is memory-bound when memory delivery limits performance before arithmetic pipelines are saturated.

Vector addition has:

```text
very little computation
large streaming reads and writes
almost no data reuse
```

Each input value is read once, added once, and normally never reused. Adding more arithmetic units would not help if values cannot arrive faster.

The expected bottleneck is therefore memory bandwidth and memory latency, not FP32 addition throughput.

This is a hypothesis. Benchmark and profiler evidence must confirm it.

## 10. CPU Reference

A correctness reference should be simple and independent:

```text
for every i in [0, N):
    expected[i] = a[i] + b[i]
```

The CPU reference should prioritize clarity, not CPU speed.

Input generation should make mistakes visible. Useful patterns include:

```text
a[i] = a simple function of i
b[i] = a different simple function of i
```

Do not initialize `output` with the expected answer before the GPU runs. Use a recognizable sentinel if you want to detect positions the kernel did not write.

## 11. Floating-Point Validation

FP32 is not real-number arithmetic. Every operation follows finite precision and IEEE-754 behavior.

For a single addition using the same FP32 inputs, CPU and GPU results will often match exactly. A reusable validation system should still understand:

- absolute error;
- relative error;
- NaN behavior;
- positive and negative infinity;
- signed zero;
- overflow and underflow.

Important rule:

```text
NaN != NaN
```

Therefore, ordinary equality reports two NaN results as different. Validation must compare NaN classification explicitly when NaN inputs are part of the test.

Examples:

```text
finite + finite       -> finite or overflow to infinity
+infinity + finite    -> +infinity
-infinity + finite    -> -infinity
+infinity + -infinity -> NaN
NaN + anything        -> NaN
```

## 12. Asynchronous Kernel Launch

A kernel launch normally returns control to the CPU before the GPU finishes:

```text
CPU records time
CPU submits kernel
CPU records time immediately
GPU may still be executing
```

Host wall-clock timing without synchronization can measure submission overhead instead of kernel duration.

Correct timing must define a boundary that includes GPU completion.

## 13. CUDA Events

CUDA events are timestamps recorded in a CUDA stream by the GPU execution timeline.

The conceptual timing sequence is:

```text
create start and stop events
perform warm-up launches
record start event
launch vector-add kernel several times
record stop event
wait for stop event
query elapsed milliseconds
divide by iteration count
destroy events
```

Important properties:

- event recording is ordered with work in the same stream;
- elapsed event time measures work between the two records;
- allocations and input initialization should remain outside the timed region;
- H2D and D2H copies should remain outside a kernel-only benchmark;
- synchronization must occur after the stop event before reading elapsed time.

## 14. Warm-Up

The first CUDA operation can include one-time costs:

- CUDA context initialization;
- module loading;
- possible code JIT;
- memory-page setup;
- cache and clock-state changes.

Timing the first launch alone often measures setup rather than steady-state kernel behavior.

Run several untimed warm-up launches before recording benchmark events.

## 15. Repetition

A small vector-add kernel can execute faster than stable host timing resolution and can be strongly affected by launch overhead.

Repeat the kernel many times inside one timed region:

```text
average kernel time = total event time / repetitions
```

The output does not change after each identical launch, so repeated launches remain valid for timing if inputs and output addresses stay unchanged.

Record the repetition count in the report.

## 16. Effective Bandwidth

For vector addition:

```text
useful bytes = 12 * N
time seconds = milliseconds / 1000

effective bandwidth GB/s = useful bytes / time seconds / 1,000,000,000
```

Example only:

```text
N = 10,000,000
useful bytes = 120,000,000 bytes
average time = 0.001 seconds
effective bandwidth = 120 GB/s
```

Use decimal GB/s when dividing by `1e9`. If using GiB/s, divide by `1024^3` and label it accurately.

## 17. Device-To-Device Copy Baseline

A simple device copy conceptually performs:

```text
read source[i]       4 bytes
write destination[i] 4 bytes
----------------------------
useful traffic        8 bytes per element
```

Its useful-bandwidth model is therefore:

```text
copy useful bytes = 8 * N
```

Do not compare only raw latency between copy and vector addition because they use different useful-byte counts. Compare correctly defined effective bandwidth and explain the definition.

A library copy and a custom copy kernel can also have different implementation behavior. State exactly which baseline you measured.

## 18. Grid-Stride Loop

The scalar baseline launches enough threads to cover the complete problem.

A grid-stride loop lets each thread process several elements:

```text
first index = global thread index
stride = total launched threads = gridDim.x * blockDim.x
next index = current index + stride
```

Example with 128 total threads:

```text
thread 5 processes 5, 133, 261, 389, ...
thread 6 processes 6, 134, 262, 390, ...
```

If both versions generate coalesced accesses and enough total work, they can perform similarly. Consecutive lanes still access consecutive positions during each loop iteration.

The grid-stride version is useful when launch size is intentionally capped or reused across many problem sizes.

## 19. Vectorized Loads

A vectorized implementation may load several adjacent FP32 values with one wider instruction, such as a 16-byte access representing four floats.

Vectorization is safe only when:

- the address meets the required alignment;
- enough complete elements remain;
- no vector crosses the allocation boundary;
- input and output layouts agree;
- tail elements are handled separately.

`cudaMalloc` returns strongly aligned base allocations, but adding an element offset can create a misaligned address.

Example:

```text
base address aligned to 16 bytes
base + 0 floats -> aligned
base + 1 float  -> shifted by 4 bytes, not 16-byte aligned
```

A vector width of four also requires special handling when `N` is not divisible by four.

Do not implement vectorization before the scalar and grid-stride versions are correct and measured.

## 20. Block Size And Performance

Correctness should not depend on block size, provided the launch is legal and the bounds check is correct.

Try block sizes such as:

```text
50   two warps, second warp partially populated
64   two complete warps
128  four complete warps
256  eight complete warps
```

Larger blocks are not automatically faster. Performance depends on:

- enough resident warps to hide latency;
- register use;
- shared-memory use;
- memory transaction efficiency;
- scheduling and launch overhead;
- total problem size.

Vector addition usually reaches similar bandwidth across several reasonable block sizes once the grid is large enough.

## 21. Small And Large Problems

Small vectors are often dominated by:

- kernel launch overhead;
- synchronization overhead;
- event overhead;
- insufficient parallel work.

Large vectors better expose steady-state memory bandwidth but require enough memory for three vectors.

On the GB10, remember that other processes and desktop activity can affect shared memory bandwidth. Record relevant system activity when comparing runs.

## 22. Test Shapes And Their Purpose

| Shape | What it tests |
|---:|---|
| `0` | Host handling without a zero-block launch |
| `1` | Smallest nonempty vector |
| `31`, `32`, `33` | Warp boundary behavior |
| `255`, `256`, `257` | Common block boundary behavior |
| Large `N` | Throughput and effective bandwidth |
| `N` not divisible by four | Vectorized tail handling |
| Misaligned offset | Whether vectorized access assumptions are valid |
| NaN and infinity | IEEE-754 validation behavior |

Correctness tests and performance tests serve different purposes. Tiny awkward shapes expose bugs; large shapes expose bottlenecks.

## 23. Materialize The Concepts In Your Terminal

Confirm the GB10's memory and execution properties:

```bash
cd ~/llm-kernel-lab/labs/00_device_query
./lab0
```

Revisit the thread mapping with a partial warp:

```bash
cd ~/llm-kernel-lab/labs/01_kernel_launch
./lab1 70 50
```

Observe how lanes 0-31 form warp 0, threads 32-49 form a partial warp 1, and local indexes restart at global thread 50 in block 1.

After implementing the scalar vector-add kernel:

```bash
compute-sanitizer --tool memcheck ./lab2
cuobjdump --dump-ptx ./lab2 | less
cuobjdump --dump-sass ./lab2 | less
```

In PTX or SASS, look for two global loads, one floating-point addition, and one global store. Exact instruction names and compiler transformations can vary.

Use Nsight Compute only after correctness:

```bash
ncu --set basic ./lab2
```

Do not interpret one metric in isolation. Compare requested bytes, achieved bandwidth, sectors or transactions, launch geometry, and kernel duration together.

## 24. Readiness Check

Answer these before writing the complete program:

1. Why are there three vectors rather than one?
2. Which vectors require H2D copies, and which requires D2H?
3. Why is no H2D copy required for the output?
4. How many useful bytes does one FP32 vector-add element move?
5. Why is one addition per 12 useful bytes considered low arithmetic intensity?
6. Why are consecutive lane addresses desirable?
7. Why can CPU wall-clock timing measure only launch submission?
8. Where should allocations and H2D copies be placed relative to CUDA events?
9. Why is NaN validation different from ordinary equality?
10. When is a four-float vectorized access unsafe?

## 25. Answer Key

1. Two vectors supply independent inputs and one stores the result.
2. Inputs `a` and `b` move H2D; the completed output moves D2H for validation.
3. The kernel creates the output, so copying an expected answer first is unnecessary and can hide errors.
4. Two 4-byte reads plus one 4-byte write equal 12 useful bytes.
5. The GPU performs very little arithmetic compared with the amount of data delivered and stored.
6. Neighboring lanes can be served by fewer, more efficient memory transactions.
7. Kernel launch is asynchronous, so the CPU can continue before GPU completion.
8. Place them outside a kernel-only timed interval.
9. IEEE-754 defines `NaN != NaN`; compare classification rather than ordinary equality.
10. It is unsafe when the address is insufficiently aligned, fewer than four valid elements remain, or the access would cross the allocation boundary.

## 26. Ready-To-Code Statement

You are ready when this statement makes sense:

```text
Each valid thread uses the Lab 01 global index to read one contiguous FP32 value from each input and write one output; the operation moves about 12 useful bytes for one addition, so large-vector performance should be limited mainly by memory delivery; correctness is validated separately from kernel-only CUDA-event timing.
```
