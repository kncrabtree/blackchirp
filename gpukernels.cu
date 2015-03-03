#include <cuda/cuda.h>

extern "C"

__global__ void vector_add(int n, long long *a, long long *b, long long *c)
{
    int ii = blockIdx.x*blockDim.y + threadIdx.x;
    if(ii < n)
        c[ii] = a[ii] + b[ii];
}

extern "C"

__global__ void addInPlace(int n, long long *oldData, long long *newData)
{
    int ii = blockIdx.x*blockDim.y + threadIdx.x;
    if(ii < n)
        oldData[ii] += newData[ii];
}
