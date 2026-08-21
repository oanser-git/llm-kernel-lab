# Lab 14: Materialized Causal Attention

## Goal

Implement scaled dot-product causal attention as explicit stages so every tensor, reduction, and memory cost is visible.

## Contract

For each batch and head:

```text
S = Q K^T / sqrt(head_dim)
P = softmax(S + causal_mask)
O = P V
```

Begin with equal query and KV head counts, FP32 values, and a square self-attention sequence.

## Write

- Independent PyTorch reference.
- Score-matrix kernel or GEMM path.
- Correct causal masking that never allows a query to observe a future key.
- Stable row-wise softmax using lessons from lab 06.
- Probability-times-value kernel or GEMM path.
- A host pipeline with explicit workspace sizing.

## Test Shapes

Include sequence lengths `1`, values around tile or block boundaries, several heads, small and odd head dimensions, known hand-computable inputs, and strongly scaled logits.

## Benchmark

Report stage-by-stage and end-to-end latency, workspace bytes, launch count, and error. Sweep sequence length to expose quadratic score storage and work.

## Questions

1. Which intermediate tensor dominates memory as sequence length grows?
2. At exactly which score positions is the causal mask applied?
3. Why must masking occur before softmax normalization?
4. Which stages can use GEMM and which remain reductions or elementwise work?

## Complete When

Outputs and optionally probabilities match PyTorch, causal behavior is tested directly, and quadratic workspace growth is measured.
