# The CUDA Stack

CUDA is not one program or one library. It is a collection of build tools, user-space libraries, operating-system driver components, and GPU hardware.

The most important separation is:

```text
BUILD TIME

CUDA C++ source
  -> nvcc coordinates compilation
  -> PTX virtual instructions
  -> ptxas produces architecture-specific SASS
  -> host and device code are packaged into an executable

RUNTIME

Application
  -> CUDA Runtime API (libcudart)
  -> CUDA Driver API (libcuda)
  -> NVIDIA Linux kernel driver
  -> GPU executes SASS
```

## 1. CUDA Toolkit Versus NVIDIA Driver

The CUDA Toolkit and NVIDIA driver are separate installations with different responsibilities.

| Component | Example on this machine | Purpose |
|---|---|---|
| CUDA Toolkit | `/usr/local/cuda-13.0` | Builds CUDA applications and provides development libraries |
| `nvcc` | `/usr/local/cuda-13.0/bin/nvcc` | Coordinates host and device compilation |
| `ptxas` | `/usr/local/cuda-13.0/bin/ptxas` | Converts PTX into architecture-specific SASS |
| CUDA Runtime | `libcudart.so.13` | Provides convenient `cuda*` operations |
| CUDA Driver API | `libcuda.so.580.126.09` | Provides low-level `cu*` operations in user space |
| NVIDIA kernel driver | Kernel modules such as `nvidia` and `nvidia_uvm` | Performs privileged communication with GPU hardware |
| GPU | NVIDIA GB10 | Executes native `sm_121` SASS |

Changing the Toolkit does not automatically change the NVIDIA driver. Several Toolkit versions may coexist while applications use the same installed driver.

## 2. What NVCC Does

`nvcc` is a build-time command-line program. It is not part of the CUDA Runtime API.

For a source file containing both CPU code and a GPU kernel:

```cpp
int main()
{
    kernel<<<1, 32>>>();
}

__global__ void kernel()
{
    // GPU work
}
```

`nvcc` coordinates several tools:

```text
program.cu
  |
  +-> host compiler -> ARM64 CPU object code
  |
  +-> CUDA front end / NVVM -> PTX
                                  |
                                  -> ptxas -> sm_121 SASS in a cubin
  |
  -> fatbinary packages device images
  -> host linker creates the final executable
```

See the exact commands without executing them:

```bash
nvcc --dryrun -arch=sm_121 program.cu -o program
```

The output exposes tools such as `gcc`, `cudafe++`, `cicc`, `ptxas`, `fatbinary`, `nvlink`, and the final host linker.

## 3. PTX

PTX means **Parallel Thread Execution**. It is NVIDIA's documented virtual GPU instruction set.

Example:

```ptx
mov.u32       %r1, %tid.x;
mov.b32       %r2, 42;
st.global.u32 [%rd1], %r2;
```

PTX expresses operations using virtual registers and virtual GPU concepts. It is normally not executed directly by the GPU.

PTX exists to decouple programming languages and compiler front ends from physical GPU instruction sets:

```text
CUDA C++ ----\
Numba --------> PTX -> architecture-specific backend -> SASS
NVRTC --------/
```

PTX also provides a forward-compatibility path. If an application has no compatible cubin but contains compatible PTX, the user-space driver may JIT-compile that PTX for the current GPU.

## 4. SASS And Cubin

SASS is the native machine instruction set executed by NVIDIA GPU hardware.

```text
PTX
  -> ptxas at build time
  -> SASS for sm_121
  -> stored inside a cubin
```

A cubin is an ELF-format GPU binary containing SASS and metadata for a particular architecture.

PTX and SASS are different abstraction levels:

| PTX | SASS |
|---|---|
| Virtual GPU ISA | Physical GPU ISA |
| Publicly documented | Inspectable but not fully documented as a stable interface |
| Uses virtual registers | Uses physical registers and hardware instructions |
| Can be JIT-compiled | Executed by the GPU |
| More portable across architectures | Architecture-specific |

Inspect them with:

```bash
less runtime_demo.ptx
cuobjdump --dump-sass runtime_demo.cubin
```

## 5. Fat Binaries And Driver JIT

A CUDA application can contain multiple device-code images:

```text
Application executable
  ├── cubin containing SASS for sm_90
  ├── cubin containing SASS for sm_121
  └── PTX fallback
```

At runtime, the driver follows logic similar to:

```text
Compatible cubin available?
  yes -> load its SASS
  no  -> look for compatible PTX
          -> JIT-compile PTX into SASS
          -> cache and load generated code
```

The Driver API triggers module loading and possible JIT compilation. The Driver API itself is an interface; the proprietary user-space driver compiler performs the JIT work.

## 6. CUDA Runtime API

The CUDA Runtime API is the convenient, higher-level programming interface supplied by `libcudart`.

Its types and functions commonly start with `cuda`:

```cpp
cudaError_t
cudaSetDevice(...)
cudaMalloc(...)
cudaMemcpy(...)
cudaDeviceSynchronize(...)
cudaFree(...)
```

It also supports CUDA C++ launch syntax:

```cpp
kernel<<<grid, block, shared_memory, stream>>>(arguments);
```

The Runtime handles work such as primary-context management, automatic device-code registration, kernel argument preparation, and module lifetime.

The name "Runtime API" does not mean it is the only API used while a program runs. Both Runtime and Driver APIs are runtime interfaces.

## 7. CUDA Driver API

The CUDA Driver API is the lower-level user-space interface supplied by `libcuda.so.1`.

Its types and functions commonly start with `CU` or `cu`:

```cpp
CUresult
CUdevice
CUcontext
CUmodule
CUfunction

cuInit(...)
cuDeviceGet(...)
cuCtxCreate(...)
cuModuleLoad(...)
cuModuleGetFunction(...)
cuMemAlloc(...)
cuLaunchKernel(...)
cuMemcpyDtoH(...)
```

With this API, the application explicitly manages contexts, modules, function handles, allocations, launches, and cleanup.

## 8. Runtime API Versus Driver API

The two APIs can perform the same GPU work at different abstraction levels.

| Operation | Runtime API | Driver API |
|---|---|---|
| Header | `cuda_runtime.h` | `cuda.h` |
| Library | `libcudart` | `libcuda` |
| Initialize | Automatic | `cuInit` |
| Context | Runtime-managed primary context | Explicit `cuCtxCreate` or primary-context operations |
| Load kernel code | Automatically registered | `cuModuleLoad` |
| Find kernel | Compiler/runtime registration | `cuModuleGetFunction` |
| Allocate | `cudaMalloc` | `cuMemAlloc` |
| Launch | `kernel<<<...>>>()` | `cuLaunchKernel` |
| Synchronize | `cudaDeviceSynchronize` | `cuCtxSynchronize` |
| Copy | `cudaMemcpy` | `cuMemcpyDtoH`, `cuMemcpyHtoD`, and related calls |
| Free | `cudaFree` | `cuMemFree` |

The Runtime path is:

```text
Application
  -> libcudart convenience layer
  -> libcuda low-level driver operations
  -> NVIDIA kernel driver
  -> GPU
```

The direct Driver path is:

```text
Application
  -> libcuda low-level driver operations
  -> NVIDIA kernel driver
  -> GPU
```

Use the Runtime API for most CUDA C++ development. Use the Driver API when explicit module loading, runtime-generated code, detailed context control, or integration requirements justify the additional complexity.

## 9. Driver API Versus Kernel Driver

These are different components despite similar names.

```text
CUDA Driver API
  = libcuda.so.1
  = user-space library
  = callable through cu* functions

NVIDIA kernel driver
  = privileged Linux kernel modules
  = communicates with hardware
  = exposed through device nodes and system calls
```

On this machine, the real Driver API library is:

```text
/usr/lib/aarch64-linux-gnu/libcuda.so.580.126.09
```

The loaded kernel modules include:

```text
nvidia
nvidia_uvm
nvidia_modeset
nvidia_drm
```

User-space driver code communicates with them through files such as:

```text
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
```

This boundary uses operating-system operations such as `openat` and `ioctl`.

## 10. One Kernel Launch End To End

Consider:

```cpp
int *device_output = nullptr;
cudaMalloc(&device_output, sizeof(int));
write_answer<<<1, 1>>>(device_output);
cudaDeviceSynchronize();
```

The full path is approximately:

```text
1. CPU executes the application.
2. Application calls cudaMalloc in libcudart.
3. libcudart requests lower-level allocation through libcuda.
4. libcuda communicates with NVIDIA kernel modules.
5. Driver maps memory accessible by the GPU.
6. Runtime prepares the write_answer launch and arguments.
7. Driver loads matching SASS or JIT-compiles PTX.
8. Kernel driver submits GPU commands.
9. GB10 schedules the block and warp.
10. GPU executes sm_121 SASS and writes the result.
11. Synchronization reports completion to the CPU application.
```

## 11. CUDA Versions

Several version numbers describe different things:

| Value | Meaning on this machine |
|---|---|
| NVIDIA driver release | 580.126.09 |
| Maximum CUDA API compatibility reported by driver | 13.0 |
| Installed Toolkit | 13.0.88 |
| Runtime library | `libcudart.so.13.0.96` |
| GPU architecture | compute capability 12.1 / `sm_121` |

An application built with an older Runtime can generally run on a sufficiently new driver. Separately, its device code must contain compatible SASS or PTX that the driver can compile for the GPU.

The normal matched build configuration is:

```text
nvcc 13 + CUDA headers 13 + libcudart 13
  -> driver supporting CUDA 13
  -> SASS or PTX compatible with sm_121
```

## 12. What Is Public And What Is Closed

| Layer | Visibility |
|---|---|
| CUDA headers and API documentation | Public |
| PTX specification | Public and documented |
| Generated PTX | Readable text |
| Generated SASS | Inspectable with NVIDIA tools, not a stable fully documented interface |
| `nvcc`, `ptxas`, `libcudart`, `libcuda` implementations | Mostly proprietary |
| NVIDIA open GPU kernel modules | Source available for supported platforms |
| User-space driver internals, JIT compiler, firmware, hardware RTL | Proprietary |

Closed implementation does not mean its public API or generated output cannot be inspected. NVIDIA provides `cuobjdump`, `nvdisasm`, Nsight, and PTX documentation specifically for development and analysis.

## 13. Commands That Materialize The Stack

The complete executable walkthrough is in `walkthroughs/cuda_stack/README.md`.

```bash
cd ~/llm-kernel-lab/walkthroughs/cuda_stack
less README.md
```

Important commands include:

```bash
nvcc --dryrun -arch=sm_121 runtime_demo.cu -o runtime_demo
nvcc -arch=compute_121 --ptx runtime_demo.cu -o runtime_demo.ptx
nvcc -arch=sm_121 --cubin runtime_demo.cu -o runtime_demo.cubin
cuobjdump --dump-sass runtime_demo.cubin
ldd runtime_demo
LD_DEBUG=libs ./runtime_demo
lsmod
ls -l /dev/nvidia*
strace -c -e trace=ioctl ./runtime_demo
```

## Final Mental Model

```text
Toolkit builds:
CUDA C++ -> PTX -> SASS/cubin -> executable linked with Runtime support

Driver runs:
Application -> Runtime API -> Driver API -> kernel driver -> GPU

GPU executes:
SASS, not CUDA C++ and normally not PTX directly
```
