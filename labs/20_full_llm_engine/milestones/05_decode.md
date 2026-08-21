# Milestone 05: KV-Cached Decode

## Goal

Process one new token by reading previous keys and values instead of recomputing the full sequence.

## Work

- Accept one token and its absolute position.
- Compute one-token projections and RoPE.
- Append exactly one key and value position per layer.
- Use grouped-query decode attention from lab 18.
- Produce next-token logits and deterministic greedy selection.
- Reuse stable buffers and remove synchronization only when correctness permits.

## Acceptance

- Prefill followed by repeated decode matches full recomputation at every position.
- Greedy token IDs match the independent reference over several prompts.
- Cache length and capacity remain valid after every token.
- Per-token latency, tokens per second, memory traffic, and kernel launch count are reported.
