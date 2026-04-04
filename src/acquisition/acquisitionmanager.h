#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <atomic>

#include <QObject>
#include <QDateTime>
#include <QTime>
#include <QTimer>
#include <QThread>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>

#include <data/loghandler.h>
#include <data/experiment/experiment.h>

/*!
 * \brief Result of a worker-thread waveform processing batch.
 *
 * Returned by the QtConcurrent worker to the AcquisitionManager event loop
 * so that side-effect operations (advance, signals, abort) stay on the AM thread.
 */
struct FtmwProcessingResult {
    int entriesProcessed{0};
    bool success{true};
    QString errorString;
    QString warningString;
};

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

    void lifPointUpdate();
    void nextLifPoint(double delay, double frequency);
    void lifShotAcquired(int);

public slots:
    void beginExperiment(std::shared_ptr<Experiment> exp);
    void processAuxData(AuxDataStorage::AuxDataMap m);
    void processValidationData(AuxDataStorage::AuxDataMap m);
    void clockSettingsComplete(const QHash<RfConfig::ClockType,RfConfig::ClockFreq> clocks);
    void pause();
    void resume();
    void abort();

    void processLifScopeShot(const QVector<qint8> b);
    void lifHardwareReady(bool success);

private:
    std::unique_ptr<QFutureWatcher<void> > pu_fw;
    std::unique_ptr<QFutureWatcher<FtmwProcessingResult>> pu_processingWatcher;
    std::shared_ptr<Experiment> ps_currentExperiment;
    AcquisitionState d_state;
    int d_auxTimerId;
    QTimer *p_drainTimer{nullptr};
    std::atomic<bool> d_abortProcessing{false};

    void auxDataTick();
    void checkComplete();
    void finishAcquisition();
    void drainFtmwBuffer();
    void onProcessingComplete();



    // QObject interface
protected:
    void timerEvent(QTimerEvent *event) override;
};

#endif // ACQUISITIONMANAGER_H
