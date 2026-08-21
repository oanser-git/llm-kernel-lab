# Lab 16: IO-Aware Tiled Attention

## Goal

Fuse tiled score computation, online softmax, and value accumulation so causal attention does not materialize the complete score or probability matrix in global memory.

## Write

- Preserve lab 14 as the correctness and performance baseline.
- A forward attention kernel that tiles query and key/value positions.
- Running maximum, normalizer, and output accumulation derived from lab 15.
- Correct causal masking for full and partial tiles.
- Bounds handling for arbitrary sequence and head dimensions.
- A debug mode that exposes intermediate state for a tiny problem.

## Constraints

Do not call an existing FlashAttention implementation for the exercise kernel. Do not allocate `[sequence, sequence]` score or probability storage. Start with FP32 if needed, then add a low-precision storage path with FP32 state.

## Test Shapes

Reuse all lab 14 cases. Add sequence lengths and head dimensions around each selected tile boundary, all-masked tile portions, one-token sequences, and long sequences that stress online normalization.

## Benchmark

Compare output error, workspace, global-memory traffic, latency, and throughput with materialized attention. Sweep sequence length and head dimension.

## Questions

1. Which tensors remain resident in registers or shared memory for one tile?
2. How is the prior output accumulator rescaled after the running maximum changes?
3. Which tiling choice is limited by shared memory and registers?
4. Why is IO-aware attention not simply three kernels fused mechanically?

## Complete When

The implementation never materializes full scores, matches the reference, handles partial causal tiles, and demonstrates measured memory-traffic benefits.
