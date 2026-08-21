# Milestone 02: One Transformer Block

## Goal

Match every intermediate result in one transformer block before stacking the full model.

## Work

- Generate fixed input activations and reference intermediates with PyTorch.
- Implement input RMSNorm.
- Execute Q, K, and V projections with a trusted GEMM baseline.
- Apply exact RoPE and grouped-query head mapping.
- Execute causal attention without a KV-cache shortcut.
- Apply output projection and residual addition.
- Execute post-attention RMSNorm and the gated MLP.
- Apply the final residual.
- Add optional intermediate dumps for tiny fixtures.

## Acceptance

- Every operation boundary has explicit shape, dtype, layout, and stride assumptions.
- Every stage matches the independent reference within justified tolerances.
- Tests expose incorrect RoPE, head mapping, masking, normalization, and residual order.
- Model constants come from configuration rather than unexplained literals.
