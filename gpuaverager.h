#ifndef GPUAVERAGER_H
#define GPUAVERAGER_H

#include <QDataStream>
#include <QList>
#include <QVector>

#include <cuda.h>
#include <cuda_runtime_api.h>

class GpuAverager {

public:
    explicit GpuAverager();
    ~GpuAverager();

    bool initialize(const int pointsPerFrame, const int numFrames, const int bytesPerPoint, QDataStream::ByteOrder byteOrder);
    QList<QVector<qint64> > parseAndAdd(const char *newDataIn, const int shift = 0);
    QList<QVector<qint64> > parseAndRollAvg(const char *newDataIn, const qint64 currentShots, const qint64 targetShots, const int shift = 0);
    void resetAverage();
    QString getErrorString() const  { return d_errorMsg; }

private:
    int d_cudaThreadsPerBlock;
    int d_pointsPerFrame;
    int d_numFrames;
    int d_totalPoints;
    int d_bytesPerPoint;
    bool d_isLittleEndian;
    bool d_isInitialized;
    QString d_errorMsg;

    void setError(QString errMsg, cudaError_t errorCode);
    void freeMemory();

    char *p_devCharPtr = nullptr;
    char *p_hostPinnedCharPtr = nullptr;
    qint64 *p_devSumPtr = nullptr;
    qint64 *p_hostPinnedSumPtr = nullptr;
    QList<cudaStream_t> d_streamList;

};


#endif // GPUAVERAGER_H

