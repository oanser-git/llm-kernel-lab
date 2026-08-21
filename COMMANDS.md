# Command-Line Learning Log

Do not paste commands here without understanding them. For every important command, explain what each argument does and what files, processes, or GPU work it creates.

## Environment Discovery

| Command | What I expected | What happened | What I learned |
|---|---|---|---|
| `nvidia-smi` | Identify the GPU and driver | Reported NVIDIA GB10, driver 580.126.09, CUDA compatibility 13.0 | Driver CUDA compatibility is not the same thing as the installed compiler version |
| `nvcc --version` | Identify the CUDA compiler | Reported CUDA toolkit 13.0.88 | `nvcc` belongs to the toolkit and compiles `.cu` source |

## Compilation And Linking

| Command | Input | Output | Meaning of every flag |
|---|---|---|---|
| `nvcc -std=c++17 -arch=native -Xcompiler=-Wall,-Wextra opencode-device-query.cu -o lab0` | Lab 00 reviewed CUDA source | `lab0` executable | C++17; target the detected GPU; pass warning flags to the host compiler; select output name |
| `nvcc -std=c++17 -arch=native -Xcompiler=-Wall,-Wextra opencode-kernel-launch.cu -o lab1` | Lab 01 reviewed CUDA source | `lab1` executable | Compile the reviewed indexing kernel for the local GPU and enable host warnings |

## Execution And Debugging

| Command | Question answered | Result |
|---|---|---|
| `./lab0` | Can device 0 be queried? | Printed the complete GB10 report |
| `./lab0 0` | Does explicit device selection work? | Selected device 0 |
| `./lab1 257 64` | Does a rounded-up grid handle a partial final block? | Validated indexes `0-256` using five blocks |

## Correctness And Sanitizers

| Command | Bug class checked | Result |
|---|---|---|
| `./lab0 99` | Out-of-range device selection | Rejected ID 99 and printed the valid range |
| `./lab0 not-a-number` | Invalid command-line parsing | Rejected the input before calling CUDA |
| `compute-sanitizer --tool memcheck ./lab1 257 64` | Out-of-bounds and invalid GPU memory access | Completed with `ERROR SUMMARY: 0 errors` |

## Benchmarking And Profiling

| Command | Metric or hypothesis | Result |
|---|---|---|
| | | |

## Build Automation

Add this section only after you have compiled and linked programs manually and can explain which repetitive steps you are automating.
