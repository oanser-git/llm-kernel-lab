# LLM Kernel Lab

A learning map from your first CUDA program to a small LLM inference engine on the NVIDIA GB10.

This repository contains organization, written guidance, and two descriptively named CUDA files in each lab. It intentionally contains no build system or hidden implementation framework.

## One-Line Flow

```text
GPU execution and memory (00-04)
  -> transformer primitives (05-09)
  -> matrix multiplication (10-13)
  -> attention, KV cache, and sampling (14-19)
  -> first complete Qwen engine (20)
  -> runtime optimization and scaling (21-25)
  -> Llama and Gemma support (26)
```

Every phase produces something required by the next phase. The only cumulative workspace is Lab 20: later labs perform isolated experiments in their descriptively named student files, then integrate successful work back into the Lab 20 engine.

## Start Here

1. Open `ROADMAP.md` and begin with Lab 00.
2. Read that lab's `README.md` without looking for a solution elsewhere.
3. Write your implementation in the descriptive file without a prefix, such as `vector-add.cu`.
4. Discover how to compile, link, execute, debug, test, benchmark, and profile the program.
5. Record important commands and their meaning in `COMMANDS.md`.
6. Record evidence and conclusions in the lab's `REPORT.md`.
7. Mark the lab complete in `ROADMAP.md` only after you can explain it, then open the next numbered directory.

## What You Own

For every implementation lab, you create:

- all implementation inside the descriptive student `.cu` file and any additional `.cpp`, `.h`, or Python files you decide to create;
- every compiler and linker command;
- any Makefile or build system, but only when you decide you understand why it helps;
- CUDA error handling and resource cleanup;
- CPU or PyTorch references;
- correctness tests and tolerance policy;
- timing and benchmark code;
- sanitizer and profiler commands;
- optimization versions and performance analysis.

Start with direct command-line compilation. Introduce automation only when repetitive work gives you a concrete reason for it.

## Student And OpenCode Files

Every lab uses this convention:

```text
<topic>.cu              your implementation; OpenCode does not overwrite it
opencode-<topic>.cu     reviewed/corrected version created from your work
```

Example:

```text
vector-add.cu
opencode-vector-add.cu
```

Before review, the `opencode-` file is only a placeholder. During review, your file remains unchanged and corrections go into the `opencode-` file so both versions can be compared.

## Learning Loop

1. Derive the operation.
2. Predict its memory and compute behavior.
3. Write the simplest reference.
4. Write the simplest correct GPU implementation.
5. Test awkward and boundary shapes.
6. Learn the tool needed to inspect the current problem.
7. Measure before optimizing.
8. Change one important idea at a time.
9. Explain the result in your own words.

## Repository Map

- `ROADMAP.md`: ordered curriculum and progress checklist.
- `COMMANDS.md`: your command-line discoveries and explanations.
- `JOURNAL.md`: chronological learning log and unresolved questions.
- `CUDA_STACK.md`: conceptual guide to NVCC, PTX, SASS, Runtime API, Driver API, kernel driver, and GPU execution.
- `labs/00_*` through `labs/19_*`: GPU foundations and all operators needed for autoregressive inference.
- `labs/20_full_llm_engine/`: the first complete Qwen text-generation engine.
- `labs/21_*` through `labs/25_*`: sequential engine optimization and scaling labs.
- `labs/26_model_adapters/`: final Qwen, Llama, and Gemma integration.
- `walkthroughs/cuda_stack/`: terminal-level inspection of NVCC, PTX, SASS, Runtime API, Driver API, kernel modules, and GPU execution.

## Rules

- Do not copy a completed kernel before producing your own baseline.
- Do not use a helper that hides behavior you cannot yet explain.
- Do not optimize before correctness.
- Do not claim a speedup without documenting shapes, dtype, warm-up, repetitions, synchronization, and baseline.
- Compare with vendor libraries, but distinguish using a library from implementing an algorithm.
- Keep failed experiments and surprising profiler results in the report.
