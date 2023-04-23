#ifndef GPUAVERAGER_H
#define GPUAVERAGER_H

#include <QDataStream>
#include <QList>
#include <QVector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <data/experiment/digitizerconfig.h>
#include <data/experiment/fid.h>

class GpuAverager {

public:
    explicit GpuAverager();
    ~GpuAverager();

    bool initialize(DigitizerConfig *cfg);
    QVector<QVector<qint64> > parseAndAdd(const char *newDataIn, const int shift = 0);
    QVector<QVector<qint64> > parseAndRollAvg(const char *newDataIn, const quint64 currentShots, const quint64 targetShots, const int shift = 0);
    void resetAverage();
    void setCurrentData(const FidList l);
    QString getErrorString() const  { return d_errorMsg; }

private:
    DigitizerConfig *p_config{nullptr};
    int d_cudaThreadsPerBlock{1024};
    int d_totalPoints{0};
    int d_numTotalBlocks{0};
    int d_numRecordBlocks{0};
    bool d_isInitialized{false};
    QString d_errorMsg;

    void setError(QString errMsg, cudaError_t errorCode);

    char *p_devCharPtr{nullptr};
    char *p_hostPinnedCharPtr{nullptr};
    qint64 *p_devSumPtr{nullptr};
    qint64 *p_hostPinnedSumPtr{nullptr};
    QVector<cudaStream_t> d_streamList;

};


#endif // GPUAVERAGER_H

