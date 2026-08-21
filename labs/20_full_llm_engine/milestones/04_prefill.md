# Milestone 04: Prefill

## Goal

Process all prompt positions and populate a contiguous KV cache while preserving full-model numerical parity.

## Work

- Define cache layout, dtype, length, and capacity explicitly.
- Store each layer's post-RoPE keys and values at the correct positions.
- Use a multi-token causal-attention path for the prompt.
- Return final-position logits while retaining valid cache state.
- Separate reusable allocations from per-request logical state.

## Acceptance

- Prefill logits match full-sequence recomputation.
- Extracted cache entries match reference keys and values layer by layer.
- One-token, awkward-length, maximum-capacity, and overflow cases are tested.
- Time to first token and prompt tokens per second are reported separately.
