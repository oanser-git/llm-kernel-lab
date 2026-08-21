# Lab 07: RMSNorm

## Goal

Implement the normalization used by Llama-, Qwen-, and Gemma-style transformer blocks and understand reduction-plus-elementwise fusion.

## Contract

For every row of hidden states:

```text
r = 1 / sqrt(mean_j(x[j]^2) + epsilon)
y[i] = weight[i] * x[i] * r
```

## Write

- FP32 reference with model-configurable epsilon.
- FP32 CUDA baseline.
- A fused one-block-per-row version where appropriate.
- A version that accepts FP16 or BF16 storage and accumulates squares in FP32.
- An optional fused residual-add plus RMSNorm path with an unambiguous output contract.

## Test Shapes

Use realistic and awkward hidden sizes, multiple rows, zeros, constant rows, small and large magnitudes, random weights, and multiple epsilon values.

## Benchmark

Sweep hidden size and row count. Report bytes moved, latency, bandwidth, reduction strategy, and error by dtype.

## Questions

1. Why is FP32 accumulation important for low-precision inputs?
2. Which tensors can residual fusion avoid writing and rereading?
3. Why does epsilon belong to model configuration?
4. What prevents unlimited fusion across an entire transformer block?

## Complete When

FP32 and one low-precision storage path are correct, epsilon is configurable, and the report analyzes bandwidth and fusion.
