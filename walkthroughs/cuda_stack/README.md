# CUDA Stack From The Terminal

This walkthrough makes each layer visible without a build system. Type every command manually and inspect its output.

```text
CUDA source
  -> nvcc / ptxas
  -> PTX and sm_121 SASS
  -> application
  -> libcudart (Runtime API)
  -> libcuda (Driver API)
  -> NVIDIA kernel modules through /dev/nvidia*
  -> GB10 executes SASS
```

## 1. Enter The Walkthrough

```bash
cd ~/llm-kernel-lab/walkthroughs/cuda_stack
```

There are two source files:

| File | Purpose |
|---|---|
| `runtime_demo.cu` | Uses `cuda*` Runtime API calls and launches `write_answer` on the GPU |
| `driver_probe.cpp` | Uses `cu*` Driver API calls directly, without `libcudart` |
| `driver_kernel.cu` | GPU kernel compiled separately into a cubin for explicit loading |
| `driver_launch.cpp` | Loads that cubin and launches the kernel through the Driver API |

Read both files before compiling:

```bash
less runtime_demo.cu
less driver_probe.cpp
less driver_kernel.cu
less driver_launch.cpp
```

## 2. Locate The Toolkit

```bash
command -v nvcc ptxas cuobjdump nvdisasm
readlink -f /usr/local/cuda
readlink -f /usr/local/cuda/bin/nvcc
nvcc --version
```

On this machine, `/usr/local/cuda` selects `/usr/local/cuda-13.0`. `nvcc` and `ptxas` are Toolkit programs used during the build. They are not part of the Runtime API.

Ask `nvcc` to reveal the commands it would run without executing them:

```bash
nvcc --dryrun -std=c++17 -arch=sm_121 --cudart=shared \
  runtime_demo.cu -o runtime_demo_dryrun
```

Find these stages in the output:

| Program | Role shown by `--dryrun` |
|---|---|
| `gcc -E` | Preprocesses host and device views of the source |
| `cudafe++` | Processes CUDA language constructs for host compilation |
| `cicc` | Compiles CUDA device code into PTX for `compute_121` |
| `ptxas` | Assembles PTX into an `sm_121` cubin containing SASS |
| `fatbinary` | Packages cubin and PTX images for embedding |
| `nvlink` | Performs CUDA device linking |
| `g++` | Links the final ARM64 host executable with `libcudart` |

This is direct evidence that `nvcc` is a build-time coordinator. `libcudart` appears only as a library passed to the final host link.

## 3. Compile A Runtime API Application

```bash
nvcc -std=c++17 -arch=sm_121 --cudart=shared runtime_demo.cu -o runtime_demo
```

| Argument | Meaning |
|---|---|
| `-std=c++17` | Compile host C++ using the C++17 language standard |
| `-arch=sm_121` | Generate native SASS for the GB10 |
| `--cudart=shared` | Dynamically link `libcudart` so it is visible with `ldd` |
| `runtime_demo.cu` | Input source |
| `-o runtime_demo` | Output executable |

Run it:

```bash
./runtime_demo
```

Expected result:

```text
GPU wrote: 42
```

The CPU allocated GPU-accessible memory through `cudaMalloc`, launched one GPU thread, copied the result back, and freed the allocation.

## 4. Materialize Every Build Product

Generate readable virtual GPU instructions:

```bash
nvcc -std=c++17 -arch=compute_121 --ptx runtime_demo.cu -o runtime_demo.ptx
```

Generate a GPU-only binary containing native SASS:

```bash
nvcc -std=c++17 -arch=sm_121 --cubin runtime_demo.cu -o runtime_demo.cubin
```

Generate a relocatable ARM host object containing bundled device code:

```bash
nvcc -std=c++17 -arch=sm_121 -c runtime_demo.cu -o runtime_demo.o
```

Identify the artifacts:

```bash
file runtime_demo.cu runtime_demo.ptx runtime_demo.cubin runtime_demo.o runtime_demo
```

The important distinction is:

```text
runtime_demo.ptx    text instructions for NVIDIA's virtual GPU ISA
runtime_demo.cubin  ELF binary for NVIDIA sm_121, containing SASS
runtime_demo.o      ARM64 host object plus embedded device information
runtime_demo        ARM64 Linux executable that launches the GPU code
```

## 5. Read The PTX

```bash
less runtime_demo.ptx
```

Important lines in the generated file include:

```text
.version 9.0
.target sm_121
.visible .entry _Z12write_answerPi
mov.u32 %r1, %ctaid.x
mov.u32 %r2, %tid.x
st.global.u32 [%rd2], %r4
```

`%ctaid.x` is the block index, `%tid.x` is the thread index, and `st.global.u32` stores 32 bits into global memory. PTX is not the final instruction stream executed by the GPU.

## 6. Read The SASS

```bash
cuobjdump --dump-sass runtime_demo.cubin | less
```

The output identifies:

```text
code for sm_121
.target sm_121
Function : _Z12write_answerPi
```

Instructions such as `S2R`, `S2UR`, `STG.E`, and `EXIT` are real GB10 machine instructions. `STG.E` performs the global-memory store. The hexadecimal words beside each instruction are its machine encoding.

The compiler may use surprising instructions to construct constants or move bits. Judge SASS by its exact semantics and dependencies, not by expecting a one-to-one translation from C++ or PTX.

Inspect device images bundled in the final executable:

```bash
cuobjdump --list-elf runtime_demo
```

## 7. See The Runtime Library

```bash
readelf -d runtime_demo | grep NEEDED
ldd runtime_demo
```

The executable declares a dependency on:

```text
libcudart.so.13
```

On this machine it resolves to:

```text
/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13.0.96
```

This is the Toolkit's CUDA Runtime implementation. We used `--cudart=shared` because `nvcc` can otherwise link the Runtime statically, making it absent from `ldd` output.

## 8. Watch The Runtime Load The Driver API

`libcuda` may be loaded dynamically, so it does not have to appear in the executable's direct `NEEDED` entries. Ask the Linux dynamic loader to print library loading:

```bash
LD_DEBUG=libs ./runtime_demo 2>&1 | grep -E 'libcudart|libcuda\.so'
```

You should see this order:

```text
/usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.13
/lib/aarch64-linux-gnu/libcuda.so.1
```

Resolve the real files:

```bash
readlink -f /usr/local/cuda/targets/sbsa-linux/lib/libcudart.so.13
readlink -f /lib/aarch64-linux-gnu/libcuda.so.1
```

The current real Driver API library is:

```text
/usr/lib/aarch64-linux-gnu/libcuda.so.580.126.09
```

## 9. Call The Driver API Directly

Compile the ordinary `.cpp` file with the host compiler:

```bash
g++ -std=c++17 \
  -I/usr/local/cuda/include \
  driver_probe.cpp \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -lcuda \
  -o driver_probe
```

The stub library is used only to satisfy the linker while building. At runtime, Linux loads the real driver-provided `libcuda.so.1`.

Inspect and run it:

```bash
readelf -d driver_probe | grep NEEDED
ldd driver_probe
./driver_probe
```

Expected output includes:

```text
Driver API version: 13.0
Driver API devices: 1
Driver API device 0: NVIDIA GB10
```

This program has a direct dependency on `libcuda.so.1` and no dependency on `libcudart`. It demonstrates this shorter path:

```text
driver_probe -> Driver API -> kernel driver -> GPU
```

## 10. Launch The Same Work Through The Driver API

The name **Runtime API** is confusing: both the Runtime API and Driver API are called while an application runs. The difference is abstraction level.

Compile the kernel as a standalone GPU module:

```bash
nvcc -std=c++17 -arch=sm_121 --cubin driver_kernel.cu -o driver_kernel.cubin
```

Compile the host application against the Driver API:

```bash
g++ -std=c++17 \
  -I/usr/local/cuda/include \
  driver_launch.cpp \
  -L/usr/local/cuda/targets/sbsa-linux/lib/stubs \
  -lcuda \
  -o driver_launch
```

Run both approaches:

```bash
./runtime_demo
./driver_launch
```

They perform the same GPU work:

```text
GPU wrote: 42
GPU wrote through Driver API: 42
```

Compare their dependencies:

```bash
ldd runtime_demo
ldd driver_launch
```

The first directly depends on `libcudart.so.13`; the second directly depends on `libcuda.so.1`.

## Side-By-Side Operations

| Operation | Runtime API | Driver API |
|---|---|---|
| Header | `cuda_runtime.h` | `cuda.h` |
| Error type | `cudaError_t` | `CUresult` |
| Function prefix | `cuda...` | `cu...` |
| Initialize driver | Automatic on first relevant call | `cuInit(0)` |
| Select device | Default device or `cudaSetDevice` | `cuDeviceGet` |
| Create context | Runtime manages a primary context | `cuCtxCreate` |
| Make GPU code available | Automatically registered from executable | `cuModuleLoad("driver_kernel.cubin")` |
| Find kernel | Compiler/runtime registration | `cuModuleGetFunction` |
| Allocate GPU memory | `cudaMalloc` | `cuMemAlloc` |
| Launch kernel | `kernel<<<grid, block>>>(...)` | `cuLaunchKernel(...)` |
| Synchronize | `cudaDeviceSynchronize` | `cuCtxSynchronize` |
| Copy device to host | `cudaMemcpy(..., cudaMemcpyDeviceToHost)` | `cuMemcpyDtoH` |
| Free GPU memory | `cudaFree` | `cuMemFree` |
| Unload module | Automatic | `cuModuleUnload` |
| Destroy context | Runtime-managed lifecycle | `cuCtxDestroy` |

The Runtime API does not replace the Driver API implementation. It performs convenient management and eventually requests lower-level driver operations. You can choose either programming interface:

```text
Easy path:
application -> Runtime API -> Driver API implementation -> kernel driver -> GPU

Manual path:
application -> Driver API -> kernel driver -> GPU
```

Do not confuse the **Driver API** (`libcuda`, user space) with the **NVIDIA kernel driver** (privileged Linux kernel modules). They are separate layers with similar names.

## 11. See The Linux Kernel Driver

Display loaded NVIDIA kernel modules:

```bash
lsmod | grep '^nvidia'
modinfo -F version nvidia
```

The installed kernel-module version is `580.126.09`, matching the user-space Driver API library.

Display the device files used to communicate with the driver:

```bash
ls -l /dev/nvidia*
```

Important nodes include:

```text
/dev/nvidia0       the GB10 device
/dev/nvidiactl     NVIDIA control interface
/dev/nvidia-uvm    unified virtual memory interface
```

Watch the application open them:

```bash
strace -f -e trace=openat ./runtime_demo 2>&1 | grep '/dev/nvidia'
```

Count the `ioctl` system calls used to communicate with the kernel driver:

```bash
strace -c -e trace=ioctl ./runtime_demo
```

The exact count is implementation- and environment-dependent. The observed run used hundreds of `ioctl` calls; the count is not a performance metric for the kernel itself.

## 12. Final Mental Model

```text
BUILD

runtime_demo.cu
  -> nvcc front end
  -> PTX
  -> ptxas
  -> sm_121 SASS in cubin
  -> ARM64 executable linked with libcudart

RUN

runtime_demo
  -> libcudart.so.13          cudaMalloc, cudaMemcpy, <<< >>> support
  -> libcuda.so.1             context/module/launch driver operations
  -> /dev/nvidia* + ioctls    Linux kernel-driver boundary
  -> nvidia kernel modules    privileged GPU control
  -> GB10                     executes SASS
```

Neither `libcudart` nor the kernel driver compiles your `.cu` source. Build-time `nvcc` and `ptxas` normally produce SASS. The user-space driver can additionally JIT-compile embedded PTX when no suitable cubin is available.
