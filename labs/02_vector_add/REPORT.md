# Lab 02 Report: Vector Addition

**Status:** All required implementation checkpoints complete; written answers remain

## Technical Model In My Own Words

## Buffer And Transfer Plan

## Bytes And Bandwidth Prediction

- Vector addition moves `12 * N` useful bytes: two FP32 reads and one FP32 write.
- A D2D copy moves `8 * N` useful bytes: one FP32 read and one FP32 write.
- At `N = 10,000,000`, each launch processes `120,000,000` useful bytes, or `120 MB`.
- Effective bandwidth is `useful bytes / average seconds / 1e9` in decimal GB/s.

## Correctness Evidence

### Student Version

- The one-thread-per-element and grid-stride kernels are correct.
- Direct access to `std::vector::data()` works on this GB10 because pageable system-memory access is supported.
- Both complete 10,000,000-element outputs are validated after each block-size experiment.
- The benchmark uses 10 warm-up launches and averages 100 timed launches.
- The grid-stride launch is capped at four blocks per SM: 192 blocks on this 48-SM GB10.
- Compute Sanitizer reported `ERROR SUMMARY: 0 errors` for the completed benchmark.
- `vector-add-explicit-memory.cu` passes device pointers to both kernels and validates both block sizes plus the D2D output.
- Both `memcheck` and `initcheck` report `ERROR SUMMARY: 0 errors` for the explicit-memory benchmark.
- Inputs are randomized from `std::random_device`, so a failing input would not be reproducible across runs.

### OpenCode Reviewed Version

- Uses the README's portable explicit allocation and H2D/kernel/D2H flow.
- Keeps independent scalar and grid-stride outputs and validates every element from both kernels.
- The grid-stride block count is configurable and independent of the problem size.
- Validated sizes `0`, `1`, `31`, `32`, `33`, `255`, `256`, `257`, and `100000`.
- A one-block, 64-thread grid-stride launch passed size `257`, proving that threads correctly process multiple elements.
- Uses 10 warm-up launches and averages 100 timed launches for both kernels and D2D.
- Compares 128- and 256-thread blocks with a 192-block grid-stride cap.
- Compute Sanitizer reported `ERROR SUMMARY: 0 errors` for the completed 10,000,000-element benchmark.

## Benchmark Results

Direct pageable-memory path, `N = 10,000,000`, 10 warm-ups, and 100 timed repetitions:

| Kernel | Block size | Grid blocks | Average time | Effective bandwidth |
|---|---:|---:|---:|---:|
| Scalar | 128 | 78,125 | 0.778005 ms | 154.241 GB/s |
| Grid-stride | 128 | 192 | 0.895099 ms | 134.063 GB/s |
| Scalar | 256 | 39,063 | 0.763709 ms | 157.128 GB/s |
| Grid-stride | 256 | 192 | 0.847769 ms | 141.548 GB/s |

Student explicit-memory benchmark from `vector-add-explicit-memory.cu`:

| Kernel | Block size | Grid blocks | Average time | Effective bandwidth |
|---|---:|---:|---:|---:|
| Scalar | 128 | 78,125 | 0.529112 ms | 226.795 GB/s |
| Grid-stride | 128 | 192 | 0.573364 ms | 209.291 GB/s |
| Scalar | 256 | 39,063 | 0.527964 ms | 227.288 GB/s |
| Grid-stride | 256 | 192 | 0.563230 ms | 213.057 GB/s |
| D2D copy | N/A | N/A | 0.366820 ms | 218.091 GB/s |

Complete OpenCode explicit-memory benchmark, `N = 10,000,000`, 10 warm-ups, and 100 timed repetitions:

| Kernel | Block size | Grid blocks | Average time | Effective bandwidth |
|---|---:|---:|---:|---:|
| Scalar | 128 | 78,125 | 0.525147 ms | 228.507 GB/s |
| Grid-stride | 128 | 192 | 0.574686 ms | 208.810 GB/s |
| Scalar | 256 | 39,063 | 0.517690 ms | 231.799 GB/s |
| Grid-stride | 256 | 192 | 0.572379 ms | 209.651 GB/s |
| D2D copy | N/A | N/A | 0.390664 ms | 204.780 GB/s |

The capped grid-stride version is slightly slower in this run, but it covers the full vector correctly with far fewer launched threads. The direct-pageable and explicit-memory results are kept in separate executables. The D2D result uses its own `8 * N` byte model and validates all copied elements. Compute Sanitizer timings are excluded because instrumentation changes performance.

## Profiler Evidence And Lessons

## Answers To README Questions
