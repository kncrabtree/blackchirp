#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include <QDateTime>
#include <QTime>
#include <QTimer>
#include <QThread>

#include <data/loghandler.h>
#include <data/experiment/experiment.h>

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
    void lifPointUpdate();
    void nextLifPoint(double delay, double frequency);
    void lifShotAcquired(int);
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
    void processLifScopeShot(const QVector<qint8> b);
    void lifHardwareReady(bool success);
#endif

private:
    std::shared_ptr<Experiment> ps_currentExperiment;
    AcquisitionState d_state;
    int d_auxTimerId;

    void auxDataTick();
    void checkComplete();
    void finishAcquisition();



    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // ACQUISITIONMANAGER_H
