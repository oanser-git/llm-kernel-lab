# Guided Labs

Complete labs in numerical order. Every directory deliberately starts with:

- `README.md`: the problem, experiments, questions, and completion gate.
- `<topic>.cu`: your implementation, named after the lab topic.
- `opencode-<topic>.cu`: the reviewed OpenCode version, initially a placeholder.
- `REPORT.md`: a place for your evidence and explanation.

Always write in the file without the `opencode-` prefix. OpenCode reviews your work by copying it into the prefixed file and making corrections there; your original remains available for comparison. Create every additional code file yourself. A first lab may need one source file; a later lab may naturally grow separate reference, kernel, test, and benchmark files.

Compile the two versions into different executables:

```bash
nvcc -arch=native vector-add.cu -o vector-add
nvcc -arch=native opencode-vector-add.cu -o opencode-vector-add
```

The second command is meaningful only after the OpenCode placeholder has been populated during review.

Lab 20 is the cumulative engine project and contains six internal milestones. Labs 21-26 then improve that same engine in strict numerical order; they do not create separate engine copies.

For each lab, discover and record:

1. How you created and edited the files from the terminal.
2. How compilation stages transform source into an executable.
3. Which compiler and linker flags are necessary and why.
4. How CUDA errors and invalid memory accesses are detected.
5. How correctness is established independently.
6. How execution time is measured without timing the wrong work.
7. Which profiler answers the current question.

## Suggested Version Names

Use names that communicate the algorithm rather than vague labels:

```text
reduce_cpu
reduce_global_baseline
reduce_shared_tree
reduce_warp_shuffle
```

Keep an older version only when it provides a meaningful correctness or performance comparison.

## Testing Layers

1. Hand-computable tiny inputs.
2. Deterministic randomized inputs.
3. Boundary and awkward dimensions from the lab README.
4. Numerical stress values.
5. NVIDIA Compute Sanitizer.
6. Performance tests after all previous layers pass.

You choose and implement the testing mechanism. The report must explain what a passing test actually proves.
