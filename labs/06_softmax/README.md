# Lab 06: Stable Softmax

## Goal

Combine maximum and sum reductions into a numerically stable row-wise softmax while minimizing global-memory traffic.

## Contract

For each row:

```text
m = max_j(x[j])
y[i] = exp(x[i] - m) / sum_j(exp(x[j] - m))
```

## Write

- Independent CPU or PyTorch reference.
- A multi-kernel baseline that makes reduction stages visible.
- A fused one-block-per-row implementation for supported widths.
- A warp-oriented implementation for short rows.
- A clear fallback for widths too large for the chosen block strategy.

## Test Shapes

Test row counts and widths around warp and block boundaries, one-element rows, very negative and positive logits, equal values, causal-mask-like negative values, NaN, and infinity according to a documented policy.

## Benchmark

Sweep row width separately from row count. Compare latency, effective bandwidth, launch count, and error against the reference.

## Questions

1. Why must the maximum be subtracted?
2. Which intermediate tensors does fusion eliminate?
3. When can one block no longer handle a row efficiently?
4. How does softmax appear inside attention and sampling?

## Complete When

All finite stress cases are stable, short and long rows have explicit strategies, and the report quantifies the benefit or cost of fusion.
