# Lab 03 Report: Memory Coalescing

**Status:** In progress - student timing checkpoint complete; profiler evidence pending

## Address-Pattern Prediction

For the contiguous scalar kernel, lane `L` reads `input[warp_start + L]` and writes
`output[warp_start + L]`. A full aligned FP32 warp requests 128 useful input bytes
from neighboring addresses and writes 128 neighboring output bytes.

## Correctness Evidence

### Student Version

- Runs and validates contiguous, offset, memory-stride, and gather transforms.
- Uses explicit device input and output buffers with complete CPU validation.
- Passed sizes `0`, `1`, `31`, `32`, `33`, `255`, `256`, `257`, and `1000`.
- Passed block sizes `50`, `64`, and `256`.
- `memcheck` and `initcheck` both reported `ERROR SUMMARY: 0 errors` for size `257`, block size `64`.
- Offset input allocation is `N + offset` while output and reference remain `N` elements.
- Passed offsets `0`, `1`, `2`, `7`, `8`, `9`, `15`, and `16` for size `257`, block size `64`.
- Offset `16` passed both `memcheck` and `initcheck` with `ERROR SUMMARY: 0 errors`.
- Strided input allocation uses `(N - 1) * memory_stride + 1` elements while output remains `N` elements.
- Passed memory strides `1`, `2`, `3`, `4`, `7`, `8`, `16`, and `32` for size `257`, block size `64`.
- Memory stride `32` passed both `memcheck` and `initcheck` with `ERROR SUMMARY: 0 errors`.
- Gather patterns cover contiguous indices, neighboring-pair permutations, and deterministic scattered indices.
- All three gather patterns passed for size `257`, block size `64`; neighboring pairs also passed sizes `1`, `32`, and `33`.
- The scattered gather passed both `memcheck` and `initcheck` with `ERROR SUMMARY: 0 errors`.

### OpenCode Reviewed Version

- Uses explicit `cudaMalloc` input and output buffers with H2D and D2H copies.
- Validates every output against `input[i] * 2 + 1` from an independent CPU reference.
- Passed sizes `0`, `1`, `31`, `32`, `33`, `255`, `256`, `257`, and `1000`.
- Passed block sizes `50`, `64`, and `256`.
- `memcheck` and `initcheck` both reported `ERROR SUMMARY: 0 errors` for size `257`, block size `64`.

## Stride And Offset Results

Offset, memory-stride, and gather correctness are complete.

Initial benchmark with `N = 10,000,000`, block size `256`, offset `2`, memory stride `2`,
scattered gather pattern `2`, 10 warm-ups, and 100 timed repetitions:

| Pattern | Useful-byte model | Average time | Effective bandwidth |
|---|---:|---:|---:|
| Contiguous | `8 * N` | 0.350308 ms | 228.370 GB/s |
| Offset 2 | `8 * N` | 0.364131 ms | 219.701 GB/s |
| Memory stride 2 | `8 * N` | 0.521253 ms | 153.476 GB/s |
| Scattered gather | `12 * N` | 2.918680 ms | 41.114 GB/s |

The offset causes a modest reduction, stride 2 causes a larger reduction, and scattered gather
is substantially slower despite producing the same number of outputs.

Automated sweep from `benchmark.py` with `N = 10,000,000`, block size `256`, 10 warm-ups,
and 100 timed repetitions:

### Offset Sweep

| Offset | Average time | Effective bandwidth |
|---:|---:|---:|
| 0 | 0.346031 ms | 231.193 GB/s |
| 1 | 0.360006 ms | 222.218 GB/s |
| 8 | 0.368230 ms | 217.255 GB/s |

### Memory-Stride Sweep

| Memory stride | Average time | Effective bandwidth |
|---:|---:|---:|
| 1 | 0.347226 ms | 230.398 GB/s |
| 2 | 0.518459 ms | 154.304 GB/s |
| 8 | 1.575970 ms | 50.762 GB/s |
| 32 | 2.420560 ms | 33.050 GB/s |

### Gather Sweep

| Gather pattern | Meaning | Average time | Effective bandwidth |
|---:|---|---:|---:|
| 0 | Contiguous | 0.526655 ms | 227.853 GB/s |
| 1 | Neighboring pairs | 0.531839 ms | 225.632 GB/s |
| 2 | Scattered | 2.958910 ms | 40.556 GB/s |

Raw CSV and plots were generated under `/tmp/opencode/lab03-results`. The script can regenerate
them after future kernel changes.

## Memory-Transaction Evidence

Pending manual Nsight Compute collection.
