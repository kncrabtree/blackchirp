#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include <QTime>
#include <QTimer>

#include "loghandler.h"
#include "experiment.h"

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
    void logMessage(const QString,const LogHandler::MessageCode = LogHandler::Normal);
    void statusMessage(const QString);
    void experimentComplete(Experiment);
    void ftmwShotAcquired(qint64);
    void beginAcquisition();
    void timeDataSignal();
    void timeData(const QList<QPair<QString,QVariant>>);

    void newFidList(QList<Fid>);

public slots:
    void beginExperiment(Experiment exp);
    void processScopeShot(const QByteArray b);
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

    void checkComplete();
    void endAcquisition();


};

#endif // ACQUISITIONMANAGER_H
