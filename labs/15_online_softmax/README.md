# Lab 15: Online Softmax

## Goal

Derive and implement softmax whose maximum and normalization are updated incrementally across input tiles.

## Derive First

For a row split into chunks, maintain a running maximum `m` and running exponential sum `l`. When a new chunk has maximum `m_new`, rescale the old sum before combining it with the new chunk. Write the exact recurrence in `REPORT.md` before coding.

## Write

- Scalar CPU online-softmax reference.
- CUDA implementation processing a row in multiple tiles.
- A way to materialize normalized output for direct comparison.
- An optional implementation that accumulates a weighted value vector instead of storing probabilities.
- Explicit handling of fully masked rows if your contract allows them.

## Test Shapes

Split identical rows using many chunk sizes and orders. Include values with large dynamic range, repeated maxima, widths around chunk boundaries, causal prefixes, and long rows.

## Benchmark

Compare ordinary multi-pass softmax and online softmax. Distinguish algorithmic memory savings from overhead on small rows.

## Questions

1. Why must the previous sum be rescaled when the running maximum increases?
2. Which state is sufficient to combine two independently processed chunks?
3. How can the recurrence incorporate a weighted sum of value vectors?
4. Which part of FlashAttention depends on this idea?

## Complete When

The recurrence is derived correctly, arbitrary chunking matches stable softmax, and the report explains how online normalization enables tiled attention.
