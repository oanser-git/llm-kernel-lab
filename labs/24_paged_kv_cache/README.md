# Lab 24: Paged KV Cache

## Goal

Replace fixed contiguous per-sequence capacity with cache blocks addressed through a page table.

Implement the indexing and allocator experiments in this lab, then replace the contiguous cache inside your Lab 20 engine without changing model outputs.

## Work

- Define logical blocks, physical blocks, page tables, and free-block ownership.
- Allocate, append across boundaries, reuse, and release blocks safely.
- Modify decode attention to follow page-table indirection.
- Detect exhausted capacity, stale ownership, and invalid block mappings.
- Compare metadata and indirection overhead with the contiguous cache.

## Acceptance

- Logical cache contents match the contiguous implementation.
- Boundary, fragmentation, reuse, and out-of-capacity cases pass.
- One sequence cannot read another sequence's unassigned blocks.
- Memory savings and decode overhead are measured across variable lengths.

## Questions

1. Which metadata lookup maps a logical token position to a physical cache address?
2. How do block size and head layout affect fragmentation and memory coalescing?
3. Which ownership rules prevent one sequence from reading another sequence's cache?
4. When does paging overhead outweigh saved capacity?
