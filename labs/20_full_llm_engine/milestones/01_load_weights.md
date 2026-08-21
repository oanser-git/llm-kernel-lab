# Milestone 01: Configuration And Weights

## Goal

Load one exact small dense Qwen checkpoint into named tensors without silently changing dtype, dimensions, layout, or values.

## Work

- Record the exact checkpoint revision and license.
- Inspect and parse required model configuration fields.
- Validate head counts, dimensions, vocabulary size, layer count, RoPE settings, RMSNorm epsilon, and embedding tying.
- Understand the SafeTensors header, tensor metadata, byte offsets, dtypes, and file bounds.
- Load only the dtypes needed by the chosen checkpoint.
- Map every expected tensor name to an exact shape.
- Reject missing, duplicate, truncated, overlapping, and unexpected tensor data.
- Print a deterministic model inventory without dumping complete tensors.

## Acceptance

- Names, dtypes, shapes, byte ranges, and selected checksums match an independent Python inspection.
- Malformed synthetic files fail clearly.
- No GPU allocation occurs until configuration and metadata are valid.
- No weight is transposed, fused, quantized, or renamed silently.
