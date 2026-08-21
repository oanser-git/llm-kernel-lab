# Lab 26: Qwen, Llama, And Gemma Adapters

## Goal

Support additional dense text checkpoints by modeling verified architecture differences without scattering model-name conditionals through the runtime.

The Lab 20 Qwen path is the baseline. Add Llama first, then Gemma, while keeping one tested engine rather than copying it into three implementations.

## Compare First

Document exact differences in tensor names, projection bias, normalization, embedding scaling, RoPE, attention patterns, activation, logit scaling or soft-capping, weight tying, tokenizer behavior, and chat templates.

## Work

- Add one identified Llama checkpoint using only interface changes justified by real differences.
- Add one identified Gemma checkpoint after Qwen and Llama remain correct.
- Parameterize shared kernels only where behavior is genuinely shared.
- Create model-specific reference fixtures and parity tests.

## Acceptance

- One checkpoint from each family matches reference logits and greedy tokens.
- Existing Qwen behavior remains unchanged.
- Unsupported options fail with actionable errors.
- Every adapter difference is covered by a test or configuration assertion.

## Questions

1. Which behaviors are true architecture differences rather than checkpoint configuration values?
2. Where should model-specific tensor naming end and model-independent execution begin?
3. Which kernels can remain shared through parameters without becoming unreadable?
4. How will tests prove that adding one family did not change another?
