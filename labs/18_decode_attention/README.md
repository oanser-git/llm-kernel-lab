# Lab 18: Decode Attention And GQA

## Goal

Implement the one-new-query attention workload used during autoregressive decode, including grouped-query attention and variable sequence lengths.

## Contract

Each sequence contributes one new query position. Query heads map to fewer or equal KV heads. The kernel attends only through that sequence's valid cache length and returns one output vector per query head.

## Write

- PyTorch reference with explicit query-head to KV-head mapping.
- FP32 baseline reading the cache layout selected in lab 17.
- Stable online softmax over each valid history.
- Variable cache lengths across a batch.
- GQA where `query_heads / kv_heads` is validated as an integer grouping.
- A low-precision cache path after FP32 correctness.

## Test Shapes

Include history lengths `1`, warp boundaries, long histories, different lengths within one batch, MHA, MQA, GQA, odd head dimensions, repeated KV heads, and hand-computable tiny cases.

## Benchmark

Sweep history length, batch size, query/KV head ratio, and head dimension. Report latency per generated token and effective KV-cache bandwidth. Compare with prefill attention behavior.

## Questions

1. Why is decode attention commonly bandwidth- or latency-bound?
2. Which work is shared by query heads in the same GQA group?
3. How does batch size change available parallelism?
4. When does splitting one head across several blocks become useful?

## Complete When

MHA, MQA, and GQA match the reference for variable lengths, and the report identifies the decode bottleneck using profiler evidence.
