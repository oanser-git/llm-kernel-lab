# NCU Cheat Sheet

Run these commands from `labs/03_memory_coalescing`.

## Setup

```bash
NCU=/usr/local/cuda/bin/ncu
BIN=./memory-coalescing
mkdir -p lab03-ncu
```

## Compile With Source-Line Information

```bash
nvcc -std=c++17 -arch=native -lineinfo memory-coalescing.cu -o "$BIN"
```

## Program Arguments

```text
$BIN N BLOCK_SIZE OFFSET STRIDE GATHER_PATTERN WARMUPS REPETITIONS
```

Gather patterns:

```text
0 = contiguous
1 = neighboring pairs
2 = scattered
```

Kernel names:

```text
scalar_transform_kernel
offset_scalar_transform_kernel
stride_transform_kernel
gather_kernel
```

## Reusable Profile Template

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:KERNEL_NAME \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/REPORT_NAME \
  "$BIN" N BLOCK_SIZE OFFSET STRIDE GATHER_PATTERN 1 1
```

## Contiguous Kernel

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:scalar_transform_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/contiguous \
  "$BIN" 10000000 256 0 1 0 1 1
```

## Stride 1

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:stride_transform_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/stride1 \
  "$BIN" 10000000 256 0 1 0 1 1
```

## Stride 8

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:stride_transform_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/stride8 \
  "$BIN" 10000000 256 0 8 0 1 1
```

## Offset 1

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:offset_scalar_transform_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/offset1 \
  "$BIN" 10000000 256 1 1 0 1 1
```

## Contiguous Gather

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:gather_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/gather0 \
  "$BIN" 10000000 256 0 1 0 1 1
```

## Scattered Gather

```bash
sudo "$NCU" \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section SourceCounters \
  --kernel-name regex:gather_kernel \
  --launch-count 1 \
  --force-overwrite \
  --export lab03-ncu/gather2 \
  "$BIN" 10000000 256 0 1 2 1 1
```

## Read A Saved Report

Summary:

```bash
"$NCU" --import lab03-ncu/stride8.ncu-rep --page details
```

All tables and recommendations:

```bash
"$NCU" --import lab03-ncu/stride8.ncu-rep --page details --print-details all
```

Raw metric names and values:

```bash
"$NCU" --import lab03-ncu/stride8.ncu-rep --page raw
```

GUI:

```bash
ncu-ui lab03-ncu/stride8.ncu-rep
```

## Discover Available Options

```bash
"$NCU" --list-sections
"$NCU" --list-sets
"$NCU" --query-metrics
"$NCU" --help
```

## Important Parameters

```text
--kernel-name regex:NAME    profile only matching kernels
--launch-count 1            profile one matching launch
--section NAME              collect one analysis section
--export PATH               save a .ncu-rep report
--force-overwrite           replace an existing report
--import PATH               read a saved report
--page details              print section results
--print-details all         print section tables and recommendations
--page raw                  print raw metric names and values
```

Use CUDA-event timings for benchmark results. Use NCU for sectors, caches, occupancy, and stalls.
