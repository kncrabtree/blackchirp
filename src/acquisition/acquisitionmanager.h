#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include <QDateTime>
#include <QTime>
#include <QTimer>
#include <QThread>

#include <data/loghandler.h>
#include <data/experiment/experiment.h>

#ifdef BC_MOTOR
#include <modules/motor/data/motorscan.h>
#endif

class AcquisitionManager : public QObject
{
    Q_OBJECT
public:
    explicit AcquisitionManager(QObject *parent = nullptr);
    ~AcquisitionManager();

    enum AcquisitionState
    {
        Idle,
        Acquiring,
        Paused
    };

signals:
    void logMessage(QString,LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(QString,int=0);
    void experimentComplete();
    void ftmwUpdateProgress(int);
    void newClockSettings(QHash<RfConfig::ClockType,RfConfig::ClockFreq>);
    void beginAcquisition();
    void endAcquisition();
    void auxDataSignal();
    void auxData(AuxDataStorage::AuxDataMap,QDateTime);
    void motorRest();

    void takeSnapshot(std::shared_ptr<Experiment>);
    void doFinalSave(std::shared_ptr<Experiment>);
    void backupComplete();

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
    void beginExperiment(std::shared_ptr<Experiment> exp);
    void processFtmwScopeShot(const QByteArray b);
    void processAuxData(AuxDataStorage::AuxDataMap m);
    void processValidationData(AuxDataStorage::AuxDataMap m);
    void clockSettingsComplete(const QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);
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
    std::shared_ptr<Experiment> ps_currentExperiment;
    AcquisitionState d_state;
    int d_currentShift;
    float d_lastFom;
    int d_auxTimerId;

    void auxDataTick();
    void checkComplete();
    void finishAcquisition();
    bool calculateShift(const QByteArray b);
    bool scoreChirp(const QByteArray b);
    float calculateFom(const QVector<qint64> vec, const Fid fid, QPair<int,int> range, int trialShift);
    double calculateChirpRMS(const QVector<qint64> chirp, double sf, qint64 shots = 1);



#ifdef BC_MOTOR
    bool d_waitingForMotor;
#endif



    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // ACQUISITIONMANAGER_H
