#ifndef ACQUISITIONMANAGER_H
#define ACQUISITIONMANAGER_H

#include <QObject>
#include "loghandler.h"
#include "experiment.h"
#include <QTime>
#include <QTimer>

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
    void ftmwShotAcquired(int);
    void beginAcquisition();
    void timeDataSignal();

    void newFidList(QList<Fid>);

public slots:
    void beginExperiment(Experiment exp);
    void processScopeShot(const QByteArray b);
    void getTimeData();
    void processTimeData(const QList<QPair<QString,QVariant>> timeDataList);
    void pause();
    void resume();
    void abort();

private:
    Experiment d_currentExperiment;
    AcquisitionState d_state;
    QTime d_testTime;
    QTimer d_timeDataTimer;

    void checkComplete();
    void endAcquisition();


};

#endif // ACQUISITIONMANAGER_H
