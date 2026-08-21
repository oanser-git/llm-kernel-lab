# Lab 20: Baseline Full LLM Inference Engine

## Goal

Combine everything from labs 00-19 into a standalone C++/CUDA program that loads a real dense transformer checkpoint, accepts text, performs prefill and autoregressive decode, and returns generated text.

Start with one small Qwen-family base checkpoint around 0.5B parameters. Add Llama and Gemma only after the first model matches an independent reference. This lab is an inference engine using existing weights; training a foundation model is a separate later project.

## You Build Everything

- Command-line build and linking workflow.
- CUDA error handling, ownership, and memory cleanup.
- Tensor representation and allocation strategy.
- Model configuration parser.
- SafeTensors weight loader.
- Token-ID input and eventually tokenizer integration.
- Embedding lookup and every transformer layer.
- GEMM integration and your custom transformer kernels.
- KV cache, prefill, decode, and sampling.
- Correctness fixtures and PyTorch comparisons.
- Benchmarking, sanitizers, and profiling.
- Command-line text generation interface.

`full-llm-engine.cu` is intentionally almost empty. Begin there. `opencode-full-llm-engine.cu` remains the review copy. When the program becomes difficult to understand as one file, decide how to split it and record why in `REPORT.md`.

## Milestone Order

1. `milestones/01_load_weights.md`
2. `milestones/02_transformer_block.md`
3. `milestones/03_full_model.md`
4. `milestones/04_prefill.md`
5. `milestones/05_decode.md`
6. `milestones/06_text_generation.md`

After milestone 06, the engine must already work end to end for one Qwen checkpoint. Labs 21-26 then optimize and extend this same engine; do not create a second engine implementation.

## Final Acceptance

- Loads one identified Qwen checkpoint without converting its numerical values silently.
- Matches PyTorch intermediate tensors and final logits within justified tolerances.
- Produces matching greedy tokens for several prompts.
- Accepts text and returns decoded text with the correct tokenizer and chat-template policy.
- Uses a KV cache for decode rather than recomputing the complete sequence.
- Reports time to first token, prefill throughput, decode latency, tokens per second, and memory use.
- Runs without invalid accesses, races, leaks, or unhandled CUDA errors.

Labs 21-26 add streams, CUDA Graphs, quantization, paging, batching, and Llama/Gemma adapters after this baseline is correct.

## Do Not Start With

- Mixture-of-experts models.
- Multimodal models.
- Distributed inference.
- Speculative decoding.
- Every quantization format.
- A universal architecture abstraction before the second model works.
