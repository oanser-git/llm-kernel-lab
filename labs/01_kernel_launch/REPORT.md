# Lab 01 Report: Kernel Launch And Indexing

**Status:** In progress - all five core mappings complete; extensions pending

## Technical Model In My Own Words

## Mapping Prediction

For problem size 257 and block size 64:

- Blocks: `(257 + 64 - 1) / 64 = 5`
- Logical threads launched: `5 * 64 = 320`
- Valid global indexes: `0-256`
- Extra threads rejected by the bounds check: `257-319`

For problem size 70 and block size 50:

- Global thread 31 is block 0, thread 31, warp 0, lane 31.
- Global thread 32 is block 0, thread 32, warp 1, lane 0.
- Global thread 50 is block 1, thread 0, warp 0, lane 0.

## Correctness Evidence

- Validated global, block, thread, warp, and lane indexes for sizes `0`, `1`, `31`, `32`, `33`, `255`, `256`, `257`, and `1000`.
- Validated block sizes `50`, `64`, and `256`.
- Rejected block size `2048` because the GB10 limit is `1024` threads per block.
- Rejected malformed command-line input.
- Compute Sanitizer memcheck reported `0 errors` for problem size `257`, block size `64`.

## Launch Measurements

Not started. Timing belongs after all core mappings and the grid-stride version are correct.

## Profiler Evidence And Lessons

Not started.
