# Lab 08: Rotary Position Embeddings

## Goal

Apply position-dependent rotations to query and key vectors while handling layout and model conventions explicitly.

## Contract

Rotate pairs within the configured rotary dimension using supplied sine and cosine tables. Channels outside the rotary dimension remain unchanged. Batch, token, head, and channel layout must be documented.

## Write

- Independent CPU or PyTorch reference.
- Out-of-place FP32 kernel for queries and keys.
- In-place version after proving there are no read/write hazards.
- Interleaved-pair and split-half layout options, or one option plus a clear rejection of unsupported layouts.
- FP16 or BF16 storage path with suitable intermediate precision.

## Test Shapes

Include several positions, odd overall head dimensions, even rotary dimensions smaller than the head dimension, GQA with different query and KV head counts, noncontiguous logical batches represented by position IDs, and position zero.

## Benchmark

Sweep token count, head count, and head dimension. Compare precomputed sine/cosine tables with any alternative only after correctness.

## Questions

1. Why are RoPE conventions not interchangeable across model families?
2. Which dimensions should be adjacent for coalesced access?
3. When is in-place rotation safe?
4. Can RoPE be fused with another operator in prefill or decode?

## Complete When

The chosen layout is explicit, GQA shapes work, non-rotary channels are preserved, and results match the reference across positions.
