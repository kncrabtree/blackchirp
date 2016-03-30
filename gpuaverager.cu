#include "gpuaverager.h"

__global__ void initMem64_kernel(int n, long long *ptr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        ptr[i] = 0;
}

__global__ void parseAdd_kernel1byte(int numPoints, char *devNewData, long long int *devSum, int offset, int shift)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
        devSum[i] += (long long int)devNewData[i+shift];
}

__global__ void parseRollAvg_kernel1byte(int numPoints, char *devnewData, long long int *devSum, int offset, qint64 currentShots, qint64 targetShots, int shift)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPoints && i - shift < numPoints && i >= 0 && i - shift >= 0)
        devSum[i] += ((long long int)devnewData[i-shift] << 8);

    if(i < numPoints && currentShots > targetShots)
    {
        devSum[i] *= targetShots;
        devSum[i] /= currentShots;
    }

}

__global__ void parseAdd_kernel2byte(int numPoints, char *devNewData, long long int *devSum, int offset, bool le, int shift)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
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

__global__ void parseRollAvg_kernel2byte(int numPoints, char *devNewData, long long int *devSum, int offset, qint64 currentShots, qint64 targetShots, bool le, int shift)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
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

    if(i < numPoints && currentShots > targetShots)
    {
        devSum[i] *= targetShots;
        devSum[i] /= currentShots;
    }
}

GpuAverager::GpuAverager() : d_pointsPerFrame(0), d_numFrames(0), d_totalPoints(0), d_bytesPerPoint(0), d_isLittleEndian(true), d_isInitialized(false),
    p_devCharPtr(nullptr), p_hostPinnedCharPtr(nullptr), p_devSumPtr(nullptr), p_hostPinnedSumPtr(nullptr)
{
    d_cudaThreadsPerBlock = 1024;
}


GpuAverager::~GpuAverager()
{
   freeMemory();
}

void GpuAverager::freeMemory()
{
    if(p_devCharPtr != nullptr)
    {
        cudaFree(p_devCharPtr);
        p_devCharPtr = nullptr;
    }
    if(p_hostPinnedCharPtr != nullptr)
    {
        cudaFreeHost(p_hostPinnedCharPtr);
        p_hostPinnedCharPtr = nullptr;
    }
    if(p_devSumPtr != nullptr)
    {
        cudaFree(p_devSumPtr);
        p_devSumPtr = nullptr;
    }
    if(p_hostPinnedSumPtr != nullptr)
    {
        cudaFreeHost(p_hostPinnedSumPtr);
        p_hostPinnedSumPtr = nullptr;
    }

    while(!d_streamList.isEmpty())
        cudaStreamDestroy(d_streamList.takeFirst());
}

bool GpuAverager::initialize(const int pointsPerFrame, const int numFrames, const int bytesPerPoint, QDataStream::ByteOrder byteOrder)
{
    freeMemory();

    d_pointsPerFrame = pointsPerFrame;
    d_numFrames = numFrames;
    d_totalPoints = pointsPerFrame*numFrames;
    d_bytesPerPoint = bytesPerPoint;
    byteOrder == QDataStream::LittleEndian ? d_isLittleEndian = true : d_isLittleEndian = false;

    Q_ASSERT(d_pointsPerFrame > 0);
    if(d_pointsPerFrame <= 0)
    {
        d_errorMsg = QString("Could not initialize GPU. Invalid number of points per frame (%1).").arg(d_pointsPerFrame);
        return false;
    }

    Q_ASSERT(d_numFrames > 0);
    if(d_numFrames <= 0)
    {
        d_errorMsg = QString("Could not initialize GPU. Invalid number of frames (%1).").arg(d_numFrames);
        return false;
    }

    Q_ASSERT(d_bytesPerPoint > 0);
    if(d_bytesPerPoint < 1 || d_bytesPerPoint > 2)
    {
        d_errorMsg = QString("Could not initialize GPU. Invalid number of bytes per point (%1).").arg(d_bytesPerPoint);
        return false;
    }

    cudaError_t err = cudaMalloc(&p_devSumPtr,d_totalPoints*sizeof(qint64));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate GPU memory for 64 bit data."),err);
        return false;
    }

    initMem64_kernel<<<(d_totalPoints+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock>>>(d_totalPoints,p_devSumPtr);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        setError(QString("Could not initialize GPU memory to 0 for 64 bit data."),err);
        return false;
    }

    err = cudaMallocHost(&p_hostPinnedSumPtr,d_totalPoints*sizeof(qint64));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate pinned 64 bit host memory for sum."),err);
        return false;
    }

    err = cudaMalloc(&p_devCharPtr,d_totalPoints*d_bytesPerPoint*sizeof(char));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate GPU memory for character data."),err);
        return false;
    }

    err = cudaMallocHost(&p_hostPinnedCharPtr,d_totalPoints*d_bytesPerPoint*sizeof(char));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate pinned 8 bit host memory."),err);
        return false;
    }

    for(int i=0;i<d_numFrames;i++)
    {
        cudaStream_t str;
        err = cudaStreamCreate(&str);
        if(err != cudaSuccess)
        {
            setError(QString("Could not create streams for GPU processing."),err);
            return false;
        }
        else
            d_streamList.append(str);
    }

    d_isInitialized = true;
    return true;

}

QList<QVector<qint64> > GpuAverager::parseAndAdd(const char *newDataIn, const int shift)
{
    QList<QVector<qint64> > out;
    if(!d_isInitialized)
    {
        d_errorMsg = QString("Cannot process scope data because GPU was not initialized successfully.");
        return out;
    }

    //move new data to pinned memory for efficiency and for stream usage
    memcpy(p_hostPinnedCharPtr,newDataIn,d_totalPoints*d_bytesPerPoint*sizeof(char));

    //launch asynchronous streams
    for(int i=0;i<d_streamList.size();i++)
    {
        cudaMemcpyAsync(&p_devCharPtr[i*d_pointsPerFrame*d_bytesPerPoint], &p_hostPinnedCharPtr[i*d_pointsPerFrame*d_bytesPerPoint], d_pointsPerFrame*d_bytesPerPoint*sizeof(char), cudaMemcpyHostToDevice,d_streamList[i]);
        if(d_bytesPerPoint == 1)
            parseAdd_kernel1byte<<<(d_pointsPerFrame+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*d_pointsPerFrame,shift);
        else
            parseAdd_kernel2byte<<<(d_pointsPerFrame+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*d_pointsPerFrame,d_isLittleEndian,shift);
        cudaMemcpyAsync(&p_hostPinnedSumPtr[i*d_pointsPerFrame], &p_devSumPtr[i*d_pointsPerFrame], d_pointsPerFrame*sizeof(qint64), cudaMemcpyDeviceToHost,d_streamList[i]);
    }

    //while kernels and memory transfers are running, allocate memory for result data
    for(int i=0;i<d_numFrames;i++)
    {
        QVector<qint64> d(d_pointsPerFrame);
        out.append(d);
    }


    //wait for each stream to complete, then copy data from pinned memory into output vectors
    for(int i=0; i<d_streamList.size();i++)
    {
        cudaStreamSynchronize(d_streamList[i]);
        memcpy(out[i].data(),&p_hostPinnedSumPtr[i*d_pointsPerFrame],d_pointsPerFrame*sizeof(qint64));
    }

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        setError(QString("An error occured while averaging data on GPU."),err);

    return out;

}

QList<QVector<qint64> > GpuAverager::parseAndRollAvg(const char *newDataIn, const qint64 currentShots, const qint64 targetShots, const int shift)
{
    QList<QVector<qint64> > out;
    if(!d_isInitialized)
    {
        d_errorMsg = QString("Cannot process scope data because GPU was not initialized successfully.");
        return out;
    }

    //move new data to pinned memory for efficiency and for stream usage
    memcpy(p_hostPinnedCharPtr,newDataIn,d_totalPoints*d_bytesPerPoint*sizeof(char));

    //launch asynchronous streams
    for(int i=0;i<d_streamList.size();i++)
    {
        cudaMemcpyAsync(&p_devCharPtr[i*d_pointsPerFrame*d_bytesPerPoint], &p_hostPinnedCharPtr[i*d_pointsPerFrame*d_bytesPerPoint], d_pointsPerFrame*d_bytesPerPoint*sizeof(char), cudaMemcpyHostToDevice,d_streamList[i]);
        if(d_bytesPerPoint == 1)
            parseRollAvg_kernel1byte<<<(d_pointsPerFrame+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*d_pointsPerFrame,currentShots,targetShots,shift);
        else
            parseRollAvg_kernel2byte<<<(d_pointsPerFrame+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*d_pointsPerFrame,currentShots,targetShots,d_isLittleEndian,shift);
        cudaMemcpyAsync(&p_hostPinnedSumPtr[i*d_pointsPerFrame], &p_devSumPtr[i*d_pointsPerFrame], d_pointsPerFrame*sizeof(qint64), cudaMemcpyDeviceToHost,d_streamList[i]);
    }

    //while kernels and memory transfers are running, allocate memory for result data
    for(int i=0;i<d_numFrames;i++)
    {
        QVector<qint64> d(d_pointsPerFrame);
        out.append(d);
    }


    //wait for each stream to complete, then copy data from pinned memory into output vectors
    for(int i=0; i<d_streamList.size();i++)
    {
        cudaStreamSynchronize(d_streamList[i]);
        memcpy(out[i].data(),&p_hostPinnedSumPtr[i*d_pointsPerFrame],d_pointsPerFrame*sizeof(qint64));
    }

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        setError(QString("An error occured while averaging data on GPU."),err);

    return out;
}

void GpuAverager::resetAverage()
{
    initMem64_kernel<<<(d_totalPoints+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock, d_cudaThreadsPerBlock>>>(d_totalPoints,p_devSumPtr);
}

void GpuAverager::setError(QString errMsg, cudaError_t errorCode)
{
    d_errorMsg = errMsg.append(QString(" CUDA error message: %1").arg(cudaGetErrorString(errorCode)));
}
