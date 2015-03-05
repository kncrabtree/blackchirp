#include <cuda.h>
#include <cuda_runtime_api.h>
#include "fid.h"
#include <QString>

extern "C"

__global__ void vector_add(int n, long long *a, long long *b, long long *c)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < n)
        c[ii] = a[ii] + b[ii];
}

__global__ void addInPlace(int n, long long *oldData, long long *newData)
{
    int ii = blockIdx.x*blockDim.x + threadIdx.x;
    if(ii < n)
        oldData[ii] += newData[ii];
}

__global__ void initMem64_kernel(int n, long long *ptr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        ptr[i] = 0;
}

__global__ void parseAdd_kernel1byte(int numPoints, char *devNewData, long long int *devSum)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPoints)
        devSum[i] += (long long int)devNewData[i];
}

__global__ void parseAdd_kernel2byte(int numPoints, char *devNewData, long long int *devSum, bool le)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPoints)
    {
        if(le)
        {
            int16_t dat = (devNewData[2*i+1] << 8 ) | (devNewData[2*i] & 0xff);
            devSum[i] += (long long int)dat;
        }
        else
        {
            int16_t dat = (devNewData[2*i] << 8 ) | (devNewData[2*i+1] & 0xff);
            devSum[i] += (long long int)dat;
        }
    }
}

namespace GpuAvg {

//device pointers
char *devCharPtr = nullptr;
long long int *devSumPtr = nullptr;
long long int *hostPinnedSumPtr = nullptr;


cudaError_t gpuFree(void *ptr, bool host = false)
{
    if(!host)
    {
        cudaError_t err = cudaFree(ptr);
        ptr = nullptr;
        return err;
    }
    else
    {
        cudaError_t err = cudaFreeHost(ptr);
        ptr = nullptr;
        return err;
    }
}

cudaError_t gpuMalloc(void *ptr, size_t size)
{
    return cudaMalloc(&ptr,size);
}

QString initializeAcquisition(const int bytesPerPoint, const int numPoints)
{
    cudaError_t err;
    if(devSumPtr != nullptr)
    {
        err = gpuFree(devSumPtr);
        if(err != cudaSuccess)
            return QString("Could not free GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
    }

    err = cudaMalloc(&devSumPtr,numPoints*sizeof(long long int));
    if(err != cudaSuccess)
    {
        if(devSumPtr != nullptr)
            devSumPtr = nullptr;
        return QString("Could not allocate GPU memory for 64 bit data. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    initMem64_kernel<<<(numPoints+255)/256, 256>>>(numPoints,devSumPtr);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        gpuFree(devSumPtr);
        return QString("Could not initialize GPU memory to 0 for 64 bit data. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    if(hostPinnedSumPtr != nullptr)
    {
        err = cudaFreeHost(hostPinnedSumPtr);
        if(err != cudaSuccess)
        {
            gpuFree(devSumPtr);
            return QString("Could not free pinned 64 bit host memory. CUDA error message: %1").arg(cudaGetErrorString(err));
        }
    }

    err = cudaMallocHost(&hostPinnedSumPtr,numPoints*sizeof(long long int));
    if(err != cudaSuccess)
    {
        if(hostPinnedSumPtr != nullptr)
            hostPinnedSumPtr = nullptr;
        gpuFree(devSumPtr);
        return QString("Could not allocate pinned 64 bit host memory for sum. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    if(devCharPtr != nullptr)
    {
        err = gpuFree(devCharPtr);
        if(err != cudaSuccess)
        {
            gpuFree(devSumPtr);
            gpuFree(hostPinnedSumPtr,true);
            return QString("Could not free GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
        }
    }

    err = cudaMalloc(&devCharPtr,numPoints*bytesPerPoint*sizeof(char));
    if(err != cudaSuccess)
    {
        if(devCharPtr != nullptr)
            devCharPtr = nullptr;
        gpuFree(devSumPtr);
        gpuFree(hostPinnedSumPtr,true);
        return QString("Could not allocate GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
    }

    return QString();

}

int acquisitionComplete()
{
    cudaError_t err;
//    QString out;
    int out = 0;
    err = gpuFree(devSumPtr);
    if(err != cudaSuccess)
        out -=1;// QString("Could not free GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));

    err = gpuFree(devCharPtr);
    if(err != cudaSuccess)
    {
        out -=2;
//        if(out.isEmpty())
//            out = QString("Could not free GPU memory for character data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
//        else
//            out.append(QString(". Could not free GPU memory for character data. CUDA error message: %1").arg(QString(cudaGetErrorString(err))));
    }
    err = gpuFree(hostPinnedSumPtr,true);
    if(err != cudaSuccess)
        out -=3;

    return out;
}

int gpuParseAndAdd(int bytesPerPoint, int numPoints, const char *newDataIn, long long int *sumData, bool littleEndian = true)
{
    //copy new data to device, run kernel, copy sum from device
    //note: in the future, can try streams and stuff
    cudaError_t err;
    err = cudaMemcpy(devCharPtr, newDataIn, numPoints*bytesPerPoint*sizeof(char), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        return -1;//QString("Could not copy scope data to GPU. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));


    if(bytesPerPoint == 1)
        parseAdd_kernel1byte<<<(numPoints+1023)/1024, 1024>>>(numPoints,devCharPtr,devSumPtr);
    else
        parseAdd_kernel2byte<<<(numPoints+1023)/1024, 1024>>>(numPoints,devCharPtr,devSumPtr,littleEndian);

    err = cudaGetLastError();
    if(err != cudaSuccess)
        return -2;//QString("Could not parse and add scope data on GPU. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));

    err = cudaMemcpy(hostPinnedSumPtr, devSumPtr, numPoints*sizeof(long long int), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess)
        return -3;//QString("Could not copy summed data from GPU. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));

    memcpy(sumData,hostPinnedSumPtr,numPoints*sizeof(long long int));

    return 0;//QString();

}

}
