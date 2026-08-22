#pragma once

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

inline void print_impl(const char *format)
{
    std::cout << format << '\n';
}

template <typename T, typename... Args>
void print_impl(const char *format, const T &value, const Args &...args)
{
    while (*format)
    {
        if (*format == '{' && *(format + 1) == '}')
        {
            std::cout << value;
            print_impl(format + 2, args...);
            return;
        }
        std::cout << *format++;
    }
}

inline void check_cuda(cudaError_t result,
                       const char *call,
                       const char *file,
                       int line)
{
    if (result != cudaSuccess)
    {
        std::cerr << file << ':' << line << ": " << call << " failed: "
                  << cudaGetErrorName(result) << " ("
                  << cudaGetErrorString(result) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

#define PRINT(format, ...) print_impl(format, ##__VA_ARGS__)
#define CHECK_CUDA(call) check_cuda((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) CHECK_CUDA(call)
