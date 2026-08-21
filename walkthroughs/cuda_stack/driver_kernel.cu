extern "C" __global__ void write_answer(int *output)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        output[0] = 42;
    }
}
