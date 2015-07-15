#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include <QTime>
#include <QTimer>
#include <QThread>

#include "datastructs.h"
#include "experiment.h"

#ifdef BC_CUDA
#include "gpuaverager.h"
#endif

class AcquisitionManager : public QObject
{
    Q_OBJECT
public:
    explicit AcquisitionManager(QObject *parent = 0);
    ~AcquisitionManager();

    enum AcquisitionState
    {
        Idle,
        Acquiring,
        Paused
    };

signals:
    void logMessage(const QString,const BlackChirp::LogMessageCode = BlackChirp::LogNormal);
    void statusMessage(const QString);
    void experimentComplete(Experiment);
    void ftmwShotAcquired(qint64);
    void lifPointUpdate(QPair<QPoint,BlackChirp::LifPoint>);
    void nextLifPoint(double delay, double frequency);
    void lifShotAcquired(int);
    void beginAcquisition();
    void timeDataSignal();
    void timeData(const QList<QPair<QString,QVariant>>);

    void newFidList(QList<Fid>);
    void takeSnapshot(const Experiment);
    void doFinalSave(const Experiment);

public slots:
    void beginExperiment(Experiment exp);
    void processFtmwScopeShot(const QByteArray b);
    void processLifScopeShot(const LifTrace t);
    void lifHardwareReady(bool success);
    void changeRollingAverageShots(int newShots);
    void resetRollingAverage();
    void getTimeData();
    void processTimeData(const QList<QPair<QString,QVariant>> timeDataList);
    void pause();
    void resume();
    void abort();

private:
    Experiment d_currentExperiment;
    AcquisitionState d_state;
    QTime d_testTime;
    QTimer *d_timeDataTimer = nullptr;
    QThread *p_saveThread;

    void checkComplete();
    void endAcquisition();

#ifdef BC_CUDA
    GpuAverager gpuAvg;
#endif


};

#endif // ACQUISITIONMANAGER_H
