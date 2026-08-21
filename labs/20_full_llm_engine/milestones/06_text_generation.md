# Milestone 06: Text-To-Text Generation

## Goal

Turn the numerical model runtime into a complete command-line text-generation program.

## Work

- Identify the checkpoint's exact tokenizer files and special-token policy.
- First compare against trusted token IDs, then implement or integrate tokenizer behavior you understand.
- Encode input text and optional chat-template formatting explicitly.
- Run prefill and repeated decode until a stop token or length limit.
- Decode generated token IDs without corrupting byte-level or Unicode behavior.
- Support deterministic greedy generation before stochastic sampling.
- Print token IDs and logits in a debug mode.

## Acceptance

- Tokenization matches the reference on ordinary text, whitespace, punctuation, Unicode, and special tokens.
- Greedy text generation matches the reference token by token.
- Stop-token and maximum-length behavior are correct.
- The program accepts text and returns text without Python participating in model execution.
