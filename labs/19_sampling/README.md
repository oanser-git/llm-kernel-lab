# Lab 19: Greedy And Top-K Sampling

## Goal

Convert logits into deterministic next-token choices and then implement reproducible stochastic top-k sampling.

## Write

- CPU reference for argmax with an explicit tie-breaking rule.
- GPU argmax reduction over one or more batch rows.
- Top-k selection for configurable small `k`.
- Temperature handling with validation for zero or invalid values.
- Probability normalization over selected logits.
- A reproducible random-number strategy with explicit seed and sequence state.
- Optional top-p only after top-k is correct and benchmarked.

## Test Shapes

Include one vocabulary entry, realistic vocabulary sizes, duplicate maxima, all-negative logits, extreme magnitudes, several batch rows, `k=1`, `k` near vocabulary size, invalid `k`, and fixed-seed repeatability.

## Benchmark

Sweep vocabulary size, batch size, and `k`. Separate selection, normalization, and random draw costs. Compare GPU work with host transfer plus CPU sampling for batch one.

## Questions

1. Why must tie-breaking be specified for reference parity?
2. When can copying logits to the CPU be competitive?
3. Which reductions or sorting operations does top-k require?
4. How should random state advance across batched sequences?

## Complete When

Greedy results are deterministic, top-k samples are reproducible for fixed state, invalid inputs are rejected, and GPU versus CPU tradeoffs are measured.
