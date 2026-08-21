# Lab 13: Tensor Core GEMM

## Goal

Use Blackwell Tensor Core matrix-multiply-accumulate support through an appropriate CUDA interface and understand the layout, shape, and precision constraints.

## Write

- FP16 or BF16 input matrices with FP32 accumulation.
- A small WMMA, MMA, CuTe, or CUTLASS-based educational kernel chosen after checking GB10 support in the installed CUDA toolkit.
- Padding or predication for dimensions not naturally aligned to the selected instruction tile.
- FP32 reference comparison with dtype-appropriate error analysis.
- cuBLASLt comparison using the same input and accumulation policy.

## Test Shapes

Test exact instruction-tile multiples, all three dimensions around tile boundaries, small matrices, projection-like rectangular matrices, values sensitive to low precision, and non-finite values under a documented policy.

## Benchmark

Report operation count, latency, achieved throughput, selected math mode, input dtype, accumulation dtype, and library algorithm. Compare prefill-like large GEMM with decode-like small-M shapes.

## Questions

1. Which computation is performed by one matrix-multiply-accumulate instruction?
2. Why are operand layout and fragment ownership hardware-specific concerns?
3. Why can FP32 accumulation still differ from full FP32 GEMM?
4. Why does Tensor Core peak throughput not predict one-token decode speed?

## Complete When

At least one genuine Tensor Core path is verified through documentation or generated instructions, edge shapes are handled, and the comparison with cuBLASLt is numerically and methodologically fair.
