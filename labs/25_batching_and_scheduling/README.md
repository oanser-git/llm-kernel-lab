# Lab 25: Batching And Scheduling

## Goal

Execute several requests with different prompt and generation lengths while preserving each request's state and output order.

Build the scheduling experiment here, then integrate it with the Lab 20 engine and Lab 24 paged KV cache.

## Work

- Represent request lifecycle, sequence length, cache mapping, and sampling state explicitly.
- Batch compatible prefill and decode work.
- Remove completed requests and admit new work safely.
- Begin with a simple scheduler before attempting continuous batching.
- Measure latency and throughput tradeoffs across batch sizes.

## Acceptance

- Batched results match independent batch-one execution.
- Mixed lengths, early completion, capacity pressure, and deterministic ordering are tested.
- Per-request latency and aggregate throughput are both reported.
- Scheduling remains separate from model mathematics.

## Questions

1. Which request fields must remain independent while tensors are batched?
2. Which requests can share one prefill or decode launch?
3. How do throughput-oriented batches affect per-request latency?
4. What state must be released when a request completes or fails?
