# Lab 23: Weight-Only Quantization

## Goal

Compress linear-layer weights, reconstruct their values under an explicit scale convention, and measure the accuracy/bandwidth tradeoff during inference.

After isolated linear-layer validation, integrate the format into the Lab 20 engine and measure both model-level accuracy and decode performance.

## Start With INT8

Define symmetric per-channel or group-wise quantization:

```text
q = clamp(round(w / scale), q_min, q_max)
w_approx = q * scale
```

Only add packed INT4 after INT8 quantization, dequantization, and linear output are correct.

## Write

- CPU quantizer and dequantizer with explicit rounding and clipping.
- CUDA dequantization kernel.
- Reference linear layer using dequantized weights.
- Weight-only linear path that avoids writing a complete dequantized weight matrix, using a suitable custom or library-supported strategy.
- Group-size, scale layout, original dimensions, and padding metadata.
- Error analysis for weights, linear outputs, and selected downstream logits.

## Test Shapes

Include all-zero groups, constant groups, outliers, partial final groups, dimensions requiring packing padding, realistic projection shapes, small-M decode, and larger-M prefill.

## Benchmark

Report compressed bytes including scales and metadata, dequantization overhead, linear latency, throughput, and output error. Compare decode and prefill shapes separately.

## Questions

1. Why can weight-only quantization help decode more than prefill?
2. How does group size trade metadata and error?
3. Why must packing order and scale indexing be treated as part of the format?
4. When does dequantization cost erase bandwidth savings?

## Complete When

INT8 is fully correct, one fused or library-assisted weight-only linear path is measured, and any INT4 extension handles packing tails and metadata unambiguously.
