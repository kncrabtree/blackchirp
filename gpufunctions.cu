#include <cuda.h>
#include <cuda_runtime_api.h>
#include "fid.h"
#include <QString>

extern "C"

__global__ void initMem64_kernel(int n, long long *ptr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        ptr[i] = 0;
}

__global__ void parseAdd_kernel1byte(int numPoints, char *devNewData, long long int *devSum, int offset)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPoints)
        devSum[i] += (long long int)devNewData[i];
}

__global__ void parseAdd_kernel2byte(int numPoints, char *devNewData, long long int *devSum, int offset, bool le)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
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
char *hostPinnedCharPtr = nullptr;
long long int *devSumPtr = nullptr;
long long int *hostPinnedSumPtr = nullptr;
QList<cudaStream_t> streamList;

cudaError_t gpuFree(void *ptr, bool host = false)
{
    if(!host)
    {
        cudaError_t err = cudaFree(ptr);
        return err;
    }
    else
    {
        cudaError_t err = cudaFreeHost(ptr);
        return err;
    }
}

cudaError_t gpuMalloc(void *ptr, size_t size)
{
    return cudaMalloc(&ptr,size);
}

void freeStreamList()
{
    while(!streamList.isEmpty())
        cudaStreamDestroy(streamList.takeFirst());
}

cudaError_t makeStreamList(const int numStreams)
{
    bool success = true;
    cudaError_t err;
    for(int i=0;i<numStreams;i++)
    {
        cudaStream_t str;
        err = cudaStreamCreate(&str);
        Q_ASSERT(err == cudaSuccess);
        if(err != cudaSuccess)
        {
            success = false;
            break;
        }
        else
            streamList.append(str);
    }

    if(!success)
        freeStreamList();

    return err;
}

QString initializeAcquisition(const int bytesPerPoint, const int numPoints, const int numFrames)
{
    cudaError_t err;
    if(devSumPtr != nullptr)
    {
        err = gpuFree(devSumPtr);
        Q_ASSERT(err == cudaSuccess);
        devSumPtr = nullptr;
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
    Q_ASSERT(err == cudaSuccess);
    if(err != cudaSuccess)
    {
        gpuFree(devSumPtr);
        devSumPtr = nullptr;
        return QString("Could not initialize GPU memory to 0 for 64 bit data. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    if(hostPinnedSumPtr != nullptr)
    {
        err = cudaFreeHost(hostPinnedSumPtr);
        Q_ASSERT(err == cudaSuccess);
        if(err != cudaSuccess)
        {
            gpuFree(devSumPtr);
            devSumPtr = nullptr;
            return QString("Could not free pinned 64 bit host memory. CUDA error message: %1").arg(cudaGetErrorString(err));
        }
    }

    err = cudaMallocHost(&hostPinnedSumPtr,numPoints*sizeof(long long int));
    Q_ASSERT(err == cudaSuccess);
    if(err != cudaSuccess)
    {
        if(hostPinnedSumPtr != nullptr)
            hostPinnedSumPtr = nullptr;
        gpuFree(devSumPtr);
        devSumPtr = nullptr;
        return QString("Could not allocate pinned 64 bit host memory for sum. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    if(devCharPtr != nullptr)
    {
        err = gpuFree(devCharPtr);
        Q_ASSERT(err == cudaSuccess);
        if(err != cudaSuccess)
        {
            gpuFree(devSumPtr);
            devSumPtr = nullptr;
            gpuFree(hostPinnedSumPtr,true);
            hostPinnedSumPtr = nullptr;
            return QString("Could not free GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
        }
    }

    err = cudaMalloc(&devCharPtr,numPoints*bytesPerPoint*sizeof(char));
    Q_ASSERT(err == cudaSuccess);
    if(err != cudaSuccess)
    {
        if(devCharPtr != nullptr)
            devCharPtr = nullptr;
        gpuFree(devSumPtr);
        devSumPtr = nullptr;
        gpuFree(hostPinnedSumPtr,true);
        hostPinnedSumPtr = nullptr;
        return QString("Could not allocate GPU memory for 64 bit data. CUDA error message: %1").arg(QString(cudaGetErrorString(err)));
    }

    if(hostPinnedCharPtr != nullptr)
    {
        err = cudaFreeHost(hostPinnedCharPtr);
        Q_ASSERT(err == cudaSuccess);
        if(err != cudaSuccess)
        {
            gpuFree(devSumPtr);
            devSumPtr = nullptr;
            gpuFree(devCharPtr);
            devCharPtr = nullptr;
            gpuFree(hostPinnedSumPtr,true);
            hostPinnedSumPtr = nullptr;
            return QString("Could not free pinned 8 bit host memory. CUDA error message: %1").arg(cudaGetErrorString(err));
        }
    }

    err = cudaMallocHost(&hostPinnedCharPtr,numPoints*sizeof(long long int));
    Q_ASSERT(err == cudaSuccess);
    if(err != cudaSuccess)
    {
        if(hostPinnedCharPtr != nullptr)
            hostPinnedCharPtr = nullptr;
        gpuFree(devSumPtr);
        devSumPtr = nullptr;
        gpuFree(devCharPtr);
        devCharPtr = nullptr;
        gpuFree(hostPinnedSumPtr,true);
        hostPinnedSumPtr = nullptr;
        return QString("Could not allocate pinned 8 bit host memory for sum. CUDA error message: %1").arg(cudaGetErrorString(err));
    }

    err = makeStreamList(numFrames);
    Q_ASSERT(err == cudaSuccess);
    if(err != cudaSuccess)
    {
        gpuFree(devSumPtr);
        devSumPtr = nullptr;
        gpuFree(devCharPtr);
        devCharPtr = nullptr;
        gpuFree(hostPinnedSumPtr,true);
        hostPinnedSumPtr = nullptr;
        gpuFree(hostPinnedCharPtr,true);
        hostPinnedCharPtr = nullptr;
        return QString("Could not create CUDA streams for GPU averaging. CUDA error message: %1").arg(cudaGetErrorString(err));
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
        out -=4;

    err = gpuFree(hostPinnedCharPtr,true);
    if(err != cudaSuccess)
        out -=8;

    devCharPtr = nullptr;
    devSumPtr = nullptr;
    hostPinnedCharPtr = nullptr;
    hostPinnedSumPtr = nullptr;

    freeStreamList();

    return out;
}

QList<QVector<qint64> > gpuParseAndAdd(int bytesPerPoint, int numFrames, int numPointsPerFrame, const char *newDataIn, bool littleEndian = true)
{
    QList<QVector<qint64> > out;
    int numPoints = numFrames*numPointsPerFrame;

    //move new data to pinned memory for efficiency and for stream usage
    memcpy(hostPinnedCharPtr,newDataIn,numPoints*bytesPerPoint*sizeof(char));

    Q_ASSERT(numFrames == streamList.size());
    //launch asynchronous streams
    for(int i=0;i<streamList.size();i++)
    {
        cudaMemcpyAsync(&devCharPtr[i*numPointsPerFrame*bytesPerPoint], &hostPinnedCharPtr[i*numPointsPerFrame*bytesPerPoint], numPointsPerFrame*bytesPerPoint*sizeof(char), cudaMemcpyHostToDevice,streamList[i]);
        if(bytesPerPoint == 1)
            parseAdd_kernel1byte<<<(numPointsPerFrame+1023)/1024, 1024, 0, streamList[i]>>>(numPoints,devCharPtr,devSumPtr,i*numPointsPerFrame);
        else
            parseAdd_kernel2byte<<<(numPointsPerFrame+1023)/1024, 1024, 0, streamList[i]>>>(numPoints,devCharPtr,devSumPtr,i*numPointsPerFrame,littleEndian);
        cudaMemcpyAsync(&hostPinnedSumPtr[i*numPointsPerFrame], &devSumPtr[i*numPointsPerFrame], numPointsPerFrame*sizeof(long long int), cudaMemcpyDeviceToHost,streamList[i]);
    }

    //while kernels and memory transfers are running, allocate memory for result data
    for(int i=0;i<numFrames;i++)
    {
        QVector<qint64> d(numPointsPerFrame);
        out.append(d);
    }


    //wait for each stream to complete, then copy data from pinned memory into output vectors
    for(int i=0; i<streamList.size();i++)
    {
        cudaStreamSynchronize(streamList[i]);
        memcpy(out[i].data(),&hostPinnedSumPtr[i*numPointsPerFrame],numPointsPerFrame*sizeof(qint64));
    }
    return out;

}

}
