#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include <QDateTime>
#include <QTime>
#include <QTimer>
#include <QThread>

#include <data/datastructs.h>
#include <data/experiment/experiment.h>

#ifdef BC_CUDA
#include <modules/cuda/gpuaverager.h>
#endif

#ifdef BC_MOTOR
#include <modules/motor/data/motorscan.h>
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
    void experimentInitialized(const Experiment);
    void experimentComplete(const Experiment);
    void ftmwUpdateProgress(qint64);
    void ftmwNumShots(qint64);
    void newClockSettings(const RfConfig);
    void beginAcquisition();
    void endAcquisition();
    void timeDataSignal();
    void timeData(const QList<QPair<QString,QVariant>>, bool plot=true, QDateTime t = QDateTime::currentDateTime());
    void motorRest();

    void newFidList(const FtmwConfig, int);
    void newFtmwConfig(const FtmwConfig);
    void takeSnapshot(const Experiment);
    void doFinalSave(const Experiment);
    void snapshotComplete();

#ifdef BC_LIF
    void lifPointUpdate(const LifConfig);
    void nextLifPoint(double delay, double frequency);
    void lifShotAcquired(int);
#endif

#ifdef BC_MOTOR
    void startMotorMove(double x, double y, double z);
    void motorProgress(int);
    void motorDataUpdate(const MotorScan s);
#endif

public slots:
    void beginExperiment(Experiment exp);
    void processFtmwScopeShot(const QByteArray b);
    void changeRollingAverageShots(int newShots);
    void resetRollingAverage();
    void getTimeData();
    void processTimeData(const QList<QPair<QString,QVariant>> timeDataList, bool plot);
    void clockSettingsComplete();
    void pause();
    void resume();
    void abort();

#ifdef BC_LIF
    void processLifScopeShot(const LifTrace t);
    void lifHardwareReady(bool success);
#endif

#ifdef BC_MOTOR
    void motorMoveComplete(bool success);
    void motorTraceReceived(const QVector<double> dat);
#endif

private:
    Experiment d_currentExperiment;
    AcquisitionState d_state;
    QTimer *d_timeDataTimer = nullptr;
    QThread *p_saveThread;
    int d_currentShift;
    float d_lastFom;

    void checkComplete();
    void finishAcquisition();
    bool calculateShift(const QByteArray b);
    bool scoreChirp(const QByteArray b);
    float calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int,int> range, int trialShift);
    double calculateChirpRMS(const QVector<qint64> chirp, double sf, qint64 shots = 1);

#ifdef BC_CUDA
    GpuAverager gpuAvg;
#endif

#ifdef BC_MOTOR
    bool d_waitingForMotor;
#endif


};

#endif // ACQUISITIONMANAGER_H
