# Milestone 03: Full-Model Logits

## Goal

Stack all transformer blocks and match final logits for complete token sequences before implementing cached decode.

## Work

- Load token embeddings, every layer, final RMSNorm, and language-model head.
- Reuse one block implementation with layer-specific weights.
- Plan reusable activation memory instead of allocating inside every layer.
- Accept explicit token IDs from a file or command line.
- Compare selected early, middle, and final layer intermediates.
- Compare final logits and top-token ordering.

## Acceptance

- Several token patterns and sequence lengths match reference logits.
- Tied or untied output embeddings follow configuration.
- Invalid token IDs and excessive lengths fail clearly.
- Peak allocation and layer-by-layer execution time are measured.
