# Lab 22: CUDA Graphs

## Goal

Reduce repeated host launch overhead by capturing and replaying a stable GPU workload resembling one decode step.

Begin with a small isolated workload, then capture a stable decode path from the Lab 20 engine and compare it with eager execution.

## Write

- A multi-kernel eager baseline using operations completed in earlier labs.
- Stream capture into a CUDA Graph.
- Graph instantiation and repeated launch.
- Correct lifetime management for graph, executable graph, streams, events, and buffers.
- A strategy for changing input contents without changing addresses.
- An experiment with supported node-parameter updates or a documented reason to recapture.

## Test Cases

Validate first launch, many replays, changed input values, fixed and changed shapes, error cleanup, and output parity with eager execution.

## Benchmark

Measure host launch overhead, GPU execution duration, and end-to-end latency separately. Sweep the number and duration of kernels in the captured workload.

## Questions

1. Which parts of a graph must remain structurally stable?
2. Why do stable memory addresses matter to an inference runtime?
3. When is graph capture overhead amortized?
4. How do variable batch size and sequence length complicate graph reuse?

## Complete When

Graph replay matches eager output, resource cleanup is correct, and the report identifies a workload where graphs help and one where they do not.
