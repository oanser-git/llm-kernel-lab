# Lab 09: Fused SwiGLU

## Goal

Fuse the elementwise activation and gating step used in transformer MLP blocks and quantify eliminated memory traffic.

## Contract

For flattened gate and up-projection outputs:

```text
SiLU(x) = x / (1 + exp(-x))
output[i] = SiLU(gate[i]) * up[i]
```

## Write

- Independent FP32 reference.
- Separate SiLU and multiply kernels with an intermediate buffer.
- Fused scalar CUDA kernel.
- Optional vectorized and FP16/BF16 storage versions after alignment and tail handling are correct.
- A launcher that makes unsupported alignment assumptions impossible or explicit.

## Test Shapes

Include empty and one-element arrays, non-vector-width tails, negative and positive extremes, zeros, NaN/infinity policy, realistic MLP dimensions, and misaligned offsets for scalar code.

## Benchmark

Compare separate and fused implementations using equal timing boundaries. Calculate requested global-memory bytes for both paths and measure effective bandwidth.

## Questions

1. Which read and write disappear after fusion?
2. Is exponential throughput or memory bandwidth dominant at each shape?
3. How can a fast approximate activation alter model logits?
4. Why should the GEMM projections generally remain separate library operations at this stage?

## Complete When

Separate and fused versions match the reference, tails are correct, and the report reconciles measured speedup with removed traffic.
