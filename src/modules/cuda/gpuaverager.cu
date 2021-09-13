#include "gpuaverager.h"

__global__ void initMem64_kernel(int n, long long *ptr)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        ptr[i] = 0;
}

__global__ void setMem64_kernel(int n, long long *ptr, const long long *newData)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
        ptr[i] = newData[i];
}

__global__ void parseAdd_kernel1byte(int numPoints, char *devNewData, long long int *devSum, int offset, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
        devSum[i] += (long long int)devNewData[i+shift]*shots;
}

__global__ void parseRollAvg_kernel1byte(int numPoints, char *devnewData, long long int *devSum, int offset, qint64 currentShots, qint64 targetShots, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPoints && i + shift < numPoints && i >= 0 && i + shift >= 0)
        devSum[i] += (((long long int)devnewData[i+shift]*shots) << 8);

    if(i < numPoints && currentShots > targetShots)
    {
        devSum[i] *= targetShots;
        devSum[i] /= currentShots;
    }

}

__global__ void parseAdd_kernel2byte(int numPoints, char *devNewData, long long int *devSum, int offset, bool le, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
    {
        if(le)
        {
            int16_t dat = (devNewData[2*(i+shift)+1] << 8 ) | (devNewData[2*(i+shift)]);
            devSum[i] += (long long int)dat*shots;
        }
        else
        {
            int16_t dat = (devNewData[2*(i+shift)] << 8 ) | (devNewData[2*(i+shift)+1]);
            devSum[i] += (long long int)dat*shots;
        }
    }
}

__global__ void parseRollAvg_kernel2byte(int numPoints, char *devNewData, long long int *devSum, int offset, qint64 currentShots, qint64 targetShots, bool le, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
    {
        if(le)
        {
            int16_t dat = (devNewData[2*(i+shift)+1] << 8 ) | (devNewData[2*(i+shift)]);
            devSum[i] += ((long long int)dat*shots) << 8;
        }
        else
        {
            int16_t dat = (devNewData[2*(i+shift)] << 8 ) | (devNewData[2*(i+shift)+1]);
            devSum[i] += ((long long int)dat*shots) << 8;
        }
    }



    if(i < numPoints && currentShots > targetShots)
    {
        devSum[i] *= targetShots;
        devSum[i] /= currentShots;
    }
}

__global__ void parseAdd_kernel4byte(int numPoints, char *devNewData, long long int *devSum, int offset, bool le, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
    {
        if(le)
        {
            int32_t dat = (devNewData[4*(i+shift)+3] << 24 ) | (devNewData[4*(i+shift)+2] << 16 ) | (devNewData[4*(i+shift)+1] << 8 ) | (devNewData[4*(i+shift)]);
            devSum[i] += (long long int)dat*shots;
        }
        else
        {
            int16_t dat = (devNewData[4*(i+shift)] << 24 ) | (devNewData[4*(i+shift)+1] << 16 ) | (devNewData[4*(i+shift)+2] << 8 ) | (devNewData[4*(i+shift)+3]);
            devSum[i] += (long long int)dat*shots;
        }
    }
}

__global__ void parseRollAvg_kernel4byte(int numPoints, char *devNewData, long long int *devSum, int offset, qint64 currentShots, qint64 targetShots, bool le, int shift, int shots = 1)
{
    int i = offset + blockIdx.x*blockDim.x + threadIdx.x;
    if(i + shift < numPoints && i + shift >= 0)
    {
        if(le)
        {
            int32_t dat = (devNewData[4*i+3] << 24 ) | (devNewData[4*i+2] << 16 ) | (devNewData[4*i+1] << 8 ) | (devNewData[4*i]);
            devSum[i] += ((long long int)dat*shots) << 8;
        }
        else
        {
            int16_t dat = (devNewData[4*i] << 24 ) | (devNewData[4*i+1] << 16 ) | (devNewData[4*i+2] << 8 ) | (devNewData[4*i+3]);
            devSum[i] += ((long long int)dat*shots) << 8;
        }
    }



    if(i < numPoints && currentShots > targetShots)
    {
        devSum[i] *= targetShots;
        devSum[i] /= currentShots;
    }
}

GpuAverager::GpuAverager()
{
}


GpuAverager::~GpuAverager()
{
    cudaFree(p_devCharPtr);
    cudaFreeHost(p_hostPinnedCharPtr);
    cudaFree(p_devSumPtr);
    cudaFreeHost(p_hostPinnedSumPtr);

    while(!d_streamList.isEmpty())
        cudaStreamDestroy(d_streamList.takeFirst());
}

bool GpuAverager::initialize(DigitizerConfig *cfg)
{
    if(!cfg)
        return false;

    p_config = cfg;
    d_totalPoints = cfg->d_numRecords*cfg->d_recordLength;
    d_numTotalBlocks = (d_totalPoints+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock;
    d_numRecordBlocks = (cfg->d_recordLength+d_cudaThreadsPerBlock-1)/d_cudaThreadsPerBlock;

    cudaError_t err = cudaMalloc(&p_devSumPtr,d_totalPoints*sizeof(qint64));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate GPU memory for 64 bit data."),err);
        return false;
    }

    initMem64_kernel<<<d_numTotalBlocks, d_cudaThreadsPerBlock>>>(d_totalPoints,p_devSumPtr);
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

    err = cudaMalloc(&p_devCharPtr,d_totalPoints*cfg->d_bytesPerPoint*sizeof(char));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate GPU memory for character data."),err);
        return false;
    }

    err = cudaMallocHost(&p_hostPinnedCharPtr,d_totalPoints*cfg->d_bytesPerPoint*sizeof(char));
    if(err != cudaSuccess)
    {
        setError(QString("Could not allocate pinned 8 bit host memory."),err);
        return false;
    }

    for(int i=0;i<cfg->d_numRecords;i++)
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

QVector<QVector<qint64> > GpuAverager::parseAndAdd(const char *newDataIn, const int shift)
{
    QVector<QVector<qint64> > out;
    if(!p_config)
    {
        d_errorMsg = QString("Cannot process scope data because GPU was not initialized successfully.");
        return out;
    }

    auto rl = p_config->d_recordLength;
    auto bpp = p_config->d_bytesPerPoint;
    auto le = p_config->d_byteOrder == DigitizerConfig::LittleEndian;
    int shots = p_config->d_blockAverage ? p_config->d_numAverages : 1;

    //move new data to pinned memory for efficiency and for stream usage
    memcpy(p_hostPinnedCharPtr,newDataIn,d_totalPoints*bpp*sizeof(char));

    //launch asynchronous streams
    for(int i=0;i<d_streamList.size();i++)
    {
        cudaMemcpyAsync(&p_devCharPtr[i*rl*bpp], &p_hostPinnedCharPtr[i*rl*bpp], rl*bpp*sizeof(char), cudaMemcpyHostToDevice,d_streamList[i]);
        if(bpp == 1)
            parseAdd_kernel1byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,shift,shots);
        else if(bpp == 2)
            parseAdd_kernel2byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,le,shift,shots);
        else if(bpp == 4)
            parseAdd_kernel4byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,le,shift,shots);

        cudaMemcpyAsync(&p_hostPinnedSumPtr[i*rl], &p_devSumPtr[i*rl], rl*sizeof(qint64),
                cudaMemcpyDeviceToHost,d_streamList[i]);
    }

    //while kernels and memory transfers are running, allocate memory for result data
    for(int i=0;i<p_config->d_numRecords;i++)
    {
        QVector<qint64> d(rl);
        out.append(d);
    }


    //wait for each stream to complete, then copy data from pinned memory into output vectors
    for(int i=0; i<d_streamList.size();i++)
    {
        cudaStreamSynchronize(d_streamList[i]);
        memcpy(out[i].data(),&p_hostPinnedSumPtr[i*rl],rl*sizeof(qint64));
    }

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        setError(QString("An error occured while averaging data on GPU."),err);

    return out;

}

QVector<QVector<qint64> > GpuAverager::parseAndRollAvg(const char *newDataIn, const quint64 currentShots, const quint64 targetShots, const int shift)
{
    QVector<QVector<qint64> > out;
    if(!p_config)
    {
        d_errorMsg = QString("Cannot process scope data because GPU was not initialized successfully.");
        return out;
    }

    auto rl = p_config->d_recordLength;
    auto bpp = p_config->d_bytesPerPoint;
    auto le = p_config->d_byteOrder == DigitizerConfig::LittleEndian;
    auto cs = static_cast<qint64>(currentShots);
    auto ts = static_cast<qint64>(targetShots);
    int shots = p_config->d_blockAverage ? p_config->d_numAverages : 1;

    //move new data to pinned memory for efficiency and for stream usage
    memcpy(p_hostPinnedCharPtr,newDataIn,d_totalPoints*bpp*sizeof(char));

    //launch asynchronous streams
    for(int i=0;i<d_streamList.size();i++)
    {
        cudaMemcpyAsync(&p_devCharPtr[i*rl*bpp], &p_hostPinnedCharPtr[i*rl*bpp], rl*bpp*sizeof(char), cudaMemcpyHostToDevice,d_streamList[i]);
        if(bpp == 1)
            parseRollAvg_kernel1byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,cs,ts,shift,shots);
        else if(bpp == 2)
            parseRollAvg_kernel2byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,cs,ts,le,shift,shots);
        else
            parseRollAvg_kernel4byte<<<d_numRecordBlocks, d_cudaThreadsPerBlock, 0, d_streamList[i]>>>
                    (d_totalPoints,p_devCharPtr,p_devSumPtr,i*rl,cs,ts,le,shift,shots);
        cudaMemcpyAsync(&p_hostPinnedSumPtr[i*rl], &p_devSumPtr[i*rl], rl*sizeof(qint64), cudaMemcpyDeviceToHost,d_streamList[i]);
    }

    //while kernels and memory transfers are running, allocate memory for result data
    for(int i=0;i<p_config->d_numRecords;i++)
    {
        QVector<qint64> d(rl);
        out.append(d);
    }


    //wait for each stream to complete, then copy data from pinned memory into output vectors
    for(int i=0; i<d_streamList.size();i++)
    {
        cudaStreamSynchronize(d_streamList[i]);
        memcpy(out[i].data(),&p_hostPinnedSumPtr[i*rl],rl*sizeof(qint64));
    }

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
        setError(QString("An error occured while averaging data on GPU."),err);

    return out;
}

void GpuAverager::resetAverage()
{
    initMem64_kernel<<<d_numTotalBlocks, d_cudaThreadsPerBlock>>>(d_totalPoints,p_devSumPtr);
}

void GpuAverager::setCurrentData(const FidList l)
{
    if(l.isEmpty())
        initMem64_kernel<<<d_numTotalBlocks, d_cudaThreadsPerBlock>>>(d_totalPoints,p_devSumPtr);
    else
    {
        for(int i=0; i<l.size(); ++i)
        {
            auto s = l.at(i).size();
            cudaMemcpy(p_devSumPtr+i*s,l.at(i).rawData().constData(),s*sizeof(qint64),cudaMemcpyHostToDevice);
        }
    }
}

void GpuAverager::setError(QString errMsg, cudaError_t errorCode)
{
    d_errorMsg = errMsg.append(QString(" CUDA error message: %1").arg(cudaGetErrorString(errorCode)));
}
