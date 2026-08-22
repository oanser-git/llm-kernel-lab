# Lab 02: Vector Addition

## What You Are Building

Write a CUDA program that adds two FP32 vectors:

```text
output[i] = a[i] + b[i]
```

Example:

```text
a        = [1, 2, 3, 4]
b        = [10, 20, 30, 40]
output   = [11, 22, 33, 44]
```

Each valid GPU thread calculates one output element.

## Files

```text
vector-add.cu             your implementation
opencode-vector-add.cu    reserved for the reviewed version
LECTURE.md                technical knowledge for this lab
REPORT.md                 your predictions, results, and lessons
```

Write only in `vector-add.cu`. Do not edit the `opencode-` file.

## Read Before Coding

Read [`LECTURE.md`](LECTURE.md), especially these topics:

1. Three-vector data flow
2. Host-to-Device and Device-to-Host copies
3. Useful memory traffic per element
4. Arithmetic intensity and memory-bound kernels
5. CUDA event timing

## First Checkpoint: Correct Scalar Kernel

Complete only this checkpoint first. Do not implement timing, grid-stride loops, or vectorization yet.

### CPU Responsibilities

Your host code must:

1. Choose a problem size `N`.
2. Choose a block size.
3. Create host vectors `h_a`, `h_b`, `h_reference`, and `h_output`.
4. Fill `h_a` and `h_b` with known values.
5. Calculate `h_reference[i] = h_a[i] + h_b[i]` on the CPU.
6. Allocate device vectors `d_a`, `d_b`, and `d_output`.
7. Copy `h_a` to `d_a` using an H2D copy.
8. Copy `h_b` to `d_b` using an H2D copy.
9. Calculate the required grid size.
10. Launch the scalar CUDA kernel.
11. Check the launch and kernel execution for CUDA errors.
12. Copy `d_output` to `h_output` using a D2H copy.
13. Compare every element of `h_output` with `h_reference`.
14. Free all device allocations.

### GPU Kernel Responsibilities

Your scalar kernel must:

1. Calculate its global thread index using the Lab 01 formula.
2. Check whether the index is smaller than `N`.
3. Read `a[index]`.
4. Read `b[index]`.
5. Add the two FP32 values.
6. Store the result in `output[index]`.

### Memory Flow

```text
CPU                                          GPU

h_a ----------- H2D copy -----------------> d_a
h_b ----------- H2D copy -----------------> d_b

                                               d_a[index]
                                                   +
                                               d_b[index]
                                                   |
                                                   v
                                              d_output[index]

h_output <------ D2H copy ---------------- d_output

CPU validates h_output against h_reference
```

`d_output` is output-only. Do not copy the expected answer into it before launching the kernel, because that could hide a kernel that failed to write results.

## First Checkpoint Tests

Test the scalar version with:

```text
N = 0
N = 1
N = 31
N = 32
N = 33
N = 255
N = 256
N = 257
N = 1000
```

These sizes test empty input, warp boundaries, block boundaries, and a normal larger input.

Also test at least these block sizes:

```text
50    contains a partial warp
64    contains two complete warps
256   common larger block size
```

The output must be correct for every tested combination.

## Stop And Request Review

Request a review after the first checkpoint passes. At that point, your program should have:

```text
CPU reference
scalar CUDA kernel
bounds checking
H2D input copies
D2H output copy
CUDA error checking
full CPU validation
boundary-shape tests
```

Do not continue to the sections below until the scalar implementation is reviewed.

---

## Second Checkpoint: Grid-Stride Kernel

After the scalar version is correct, add a second kernel version using a grid-stride loop.

Each thread starts at its global index and advances by:

```text
stride = gridDim.x * blockDim.x
```

Both kernel versions must produce the same result for every test shape.

Compare:

```text
one-element-per-thread kernel
grid-stride-loop kernel
```

## Third Checkpoint: Timing And Bandwidth

After both kernels are correct:

1. Run untimed warm-up launches.
2. Create CUDA start and stop events.
3. Keep allocation, initialization, H2D copies, and D2H copies outside the timed region.
4. Time many repeated kernel launches.
5. Divide total event time by the repetition count.
6. Report average kernel latency.
7. Calculate effective bandwidth.
8. Compare at least two block sizes.

### Useful Byte Model

For each FP32 output element:

```text
read a[i]       4 bytes
read b[i]       4 bytes
write output[i] 4 bytes
-----------------------
useful traffic  12 bytes
```

For `N` elements:

```text
useful bytes = 12 * N
```

Calculate:

```text
effective bandwidth GB/s =
    useful bytes / average time in seconds / 1,000,000,000
```

State clearly that this is useful bandwidth, not the exact number of physical memory transactions.

## Fourth Checkpoint: Device Copy Baseline

Measure a device-to-device copy separately.

Its useful-byte model is:

```text
read source[i]       4 bytes
write destination[i] 4 bytes
----------------------------
useful traffic        8 bytes per element
```

Compare effective bandwidth rather than comparing only raw latency.

## Optional Checkpoint: Vectorized Loads

Attempt vectorization only after every previous checkpoint works.

The vectorized version must document:

- required pointer alignment;
- vector width;
- handling when `N` is not divisible by the vector width;
- handling of misaligned offsets;
- protection against out-of-bounds vector loads and stores.

This checkpoint is optional.

## Floating-Point Edge Cases

After ordinary finite values pass, test:

- random finite FP32 values;
- positive infinity;
- negative infinity;
- NaN;
- values near FP32 overflow and underflow.

Remember that `NaN != NaN`, so NaN results require classification rather than ordinary equality.

## Questions To Answer In `vector-add.cu`

Add answers as `//` comments after your implementation:

1. Why is vector addition normally memory-bound?
2. When is vectorization unsafe?
3. Why can a grid-stride loop and a one-element-per-thread kernel perform similarly?
4. Which operations belong inside a kernel-only timing interval?
5. Why are H2D and D2H copies excluded from kernel-only timing?

## Completion Criteria

Lab 02 is complete when:

- the scalar and grid-stride versions pass all correctness tests;
- CUDA errors are checked;
- Compute Sanitizer reports no memory errors;
- at least two block sizes are compared;
- kernel latency is measured with CUDA events;
- the `12 * N` bandwidth calculation is explained;
- the D2D baseline is measured using its own byte model;
- `REPORT.md` contains predictions, results, and lessons.
