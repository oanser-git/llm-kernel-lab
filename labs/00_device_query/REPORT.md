# Lab 00 Report: Device Query

**Status:** Complete

## Device Report

| Property | Observed value |
|---|---:|
| Device | NVIDIA GB10 |
| Compute capability | 12.1 |
| CUDA driver/runtime API | 13.0 / 13.0 |
| Streaming multiprocessors | 48 |
| Warp size | 32 threads |
| Maximum threads per block | 1024 |
| Maximum block dimensions | 1024 x 1024 x 64 |
| Maximum grid dimensions | 2147483647 x 65535 x 65535 |
| Maximum threads per SM | 1536 |
| Maximum resident blocks per SM | 24 |
| Maximum core clock | 2418 MHz |
| Registers per block / SM | 65536 / 65536 |
| Shared memory per block | 48 KiB |
| Opt-in shared memory per block | 99 KiB |
| Shared memory per SM | 100 KiB |
| Global memory visible to CUDA | 124610.64 MiB (121.69 GiB) |
| L2 cache | 24 MiB |
| Reported memory clock | 8533 MHz |
| Memory bus width | 256 bits |
| Unified addressing | yes |
| Managed memory | yes |
| Concurrent managed access | yes |
| Pageable memory access | yes |
| Host-memory mapping | yes |
| CUDA memory pools | yes |

`nvidia-smi` and the CUDA runtime both identify one NVIDIA GB10. `nvidia-smi` reports driver 580.126.09 and CUDA compatibility 13.0; `nvcc` reports CUDA toolkit 13.0.88.

## What The Limits Mean

- **Warp size:** Hardware schedules threads in groups of 32. Block sizes should usually contain whole warps, and branches or scattered addresses within a warp can reduce efficiency.
- **1024 threads per block:** No kernel launch may use more than this total, even when each individual block dimension is legal.
- **1536 threads and 24 blocks per SM:** These are upper limits on resident work. Registers, shared memory, architecture limits, and the selected block size can reduce actual occupancy first.
- **Registers:** A block cannot require more than 65536 registers, and all resident blocks share the SM's 65536 registers. More registers per thread can therefore reduce simultaneous blocks or cause spills.
- **Shared memory:** Ordinary blocks have 48 KiB by default and can request up to 99 KiB through opt-in configuration. Resident blocks share 100 KiB per SM, so large tiles can reduce occupancy.
- **48 SMs:** Grid-wide parallelism must expose enough independent blocks to keep all SMs busy. SM count alone does not determine throughput.
- **24 MiB L2 cache:** Reused data may avoid repeated memory traffic, but cache capacity and hit rate depend on the complete workload.
- **Clock rates and memory-bus values:** These are reported limits, not achieved application performance. Memory bandwidth is not derived because transfer-rate interpretation is platform-specific; later labs must measure effective bandwidth.
- **Compute capability 12.1:** Compilation must include code supported by this GB10 architecture. It also determines available instructions and CUDA features.

## Unified Memory

The GB10 uses a unified CPU/GPU memory system rather than conventional dedicated GPU VRAM. That is why `nvidia-smi` displays memory usage as `Not Supported` even though the CUDA runtime reports about 121.69 GiB of globally addressable memory.

Unified virtual addressing means CPU and GPU allocations can participate in one virtual address model. Managed-memory support allows the runtime and driver to manage accessibility and placement. Neither feature guarantees that every page is resident near the GPU, that access is free, or that bandwidth and latency are uniform. Placement, migration, caching, synchronization, and contention still matter.

## Theoretical Versus Measured

Maximum threads, blocks, registers, shared memory, dimensions, clocks, and bus width are ceilings or capabilities. They do not report achieved occupancy, bandwidth, instruction throughput, or latency. Those require an actual kernel, controlled timing, and profiler evidence.

## Verification

- Compiled for the local GPU using warnings and C++17.
- Ran successfully with the default device and explicit device 0.
- Rejected out-of-range device 99 with the available range.
- Rejected nonnumeric device input.
- Checked every CUDA runtime call that returns `cudaError_t`.
- Compared device identity and CUDA versions with `nvidia-smi` and `nvcc --version`.

## Surprises And Open Questions

- CUDA 13 no longer exposes core and memory clocks directly in this platform's `cudaDeviceProp`; the program queries `cudaDevAttrClockRate` and `cudaDevAttrMemoryClockRate` instead.
- Reported resource maxima do not reveal which limit will dominate a real kernel. Later labs must connect register use, shared memory, and block size to measured occupancy.
- The unified-memory architecture explains why CUDA can report total memory while `nvidia-smi` does not show a conventional VRAM usage counter.
